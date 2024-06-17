"""train model."""

import numpy as np
import timm
import torch
import torch.nn as nn
from storch.hydra_utils import to_object
from storch.nest import NeST

import utils


def main():
    config, folder, save_config = utils.init_project()

    if config.env.mm_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # alias
    cfg = config.config
    tcfg = cfg.train

    nest = NeST(
        project_folder=folder.root,
        strategy=config.env.strategy,
        mixed_precision=config.env.amp,
        grad_accum_steps=tcfg.grad_accum_steps,
        compile=config.env.compile,
    )

    # worker_init_fn, generator = trainutils.set_seeds(**config.reproduce)
    worker_init_fn, generator = None, None

    utils.set_input_size(cfg.model.model_name, cfg.data)

    # build dataset
    dcfg = cfg.data
    train, val, test = utils.build_dataset(dcfg)
    trainloader = nest.build_dataloader(
        is_train=True,
        dataset=train,
        **dcfg.loader.train,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    valloader = nest.build_dataloader(is_train=False, dataset=val, **dcfg.loader.val)
    testloader = nest.build_dataloader(is_train=False, dataset=test, **dcfg.loader.test)

    # build model
    model = nest.build_model(
        timm.create_model,
        **cfg.model,
        num_classes=train.num_classes,
        pretrained_cfg_overlay=dict(
            url='',
            custom_load=True,
            num_classes=train.num_classes,
            mean=utils.DATASET_MEAN,
            std=utils.DATASET_STD,
            hf_hub_id=cfg.hub.repo_id,
            tag='st_safebooru_1k',
        ),
    )
    model_args = utils.create_model_args(cfg.model)

    # build optimizer
    optimizer = nest.build_optimizer(**to_object(cfg.optimizer), parameters=model.parameters())

    # scheduler
    scheduler = nest.build_scheduler(optimizer, **cfg.scheduler)

    # criterion
    criterion = nn.BCEWithLogitsLoss()

    # initialize training
    wandb_args = (
        dict(
            wandb_project='timm-anime-pretrain',
            wandb_name=config.run.name,
            wandb_config=config,
        )
        if cfg.wandb.wandb_logging and not config.dryrun
        else {}
    )
    nest.initialize_training(
        tcfg.epochs * len(trainloader),
        log_file=cfg.checkpoint.log_file,
        log_interval=cfg.checkpoint.log_interval,
        ckpt_keep_last=cfg.checkpoint.keep_last,
        to_log=[
            config,
        ],
        logger_name=config.run.name,
        log_gpu_memory_at=[],
        **wandb_args,
    )

    # serialization
    mc, oc = nest.prepare_stateful(model, optimizer)
    nest.set_best_model_keeper('loss', 'minimize', mc)
    nest.register(model=mc, optimizer=oc, scheduler=scheduler)
    nest.load_latest()

    # save config after all setup is done
    save_config()

    # if dryrun return without training.
    if config.dryrun:
        return

    while not nest.is_end():
        nest.set_epoch()

        model.train()
        total, loss = 0, 0
        for batch in trainloader:
            image = batch.pop('image').to(nest.device)
            label = batch.pop('label').to(nest.device)

            output = model(image)
            batch_loss = criterion(output, label)

            nest.backward_step(
                batch_loss,
                optimizer,
                scheduler,
                model,
                clip_grad_norm=tcfg.clip_grad_norm,
                max_norm=tcfg.max_norm,
            )

            batch_size = image.size(0)
            batch_loss = batch_loss.detach()  # erase grads for memory efficiency.
            total += batch_size
            loss += batch_loss * batch_size

            nest.update(
                **{
                    'Loss/train/batch': batch_loss,
                    'Progress/lr': scheduler.get_last_lr()[0],
                }
            )

        total = nest.reduce(torch.as_tensor(total, device=nest.device))
        loss = nest.reduce(loss) / total
        nest.dry_update(**{'Loss/train': loss})

        model.eval()
        total, loss, ground_truth, predictions = 0, 0, [], []
        with torch.no_grad():
            for batch in valloader:
                image = batch.pop('image').to(nest.device)
                label = batch.pop('label').to(nest.device)

                output = model(image)
                batch_loss = criterion(output, label)

                batch_size = image.size(0)
                total += batch_size
                loss += batch_loss * batch_size

                ground_truth.append(label.cpu().numpy())
                predictions.append((output > cfg.test.threshold).float().cpu().numpy())

        total = nest.reduce(torch.as_tensor(total, device=nest.device))
        loss = nest.reduce(loss) / total

        ground_truth = nest.gather(ground_truth, dst=0)
        predictions = nest.gather(predictions, dst=0)
        if nest.is_primary():
            ground_truth = np.concatenate(ground_truth, axis=0)
            predictions = np.concatenate(predictions, axis=0)
            val_results = utils.test_classification(ground_truth, predictions)['samples avg']
            val_results = {
                f'Metric/{key}': value
                for key, value in val_results.items()
                if key in ['precision', 'recall', 'f1-score']
            }
        else:
            val_results = {}
        del ground_truth, predictions

        nest.dry_update(**{'Loss/validation': loss, **val_results})

        nest.update_best_model(loss)
        nest.save()

    nest.load_best_model()
    model.eval()
    total, loss, ground_truth, predictions = 0, 0, [], []
    with torch.no_grad():
        for batch in testloader:
            image = batch.pop('image').to(nest.device)
            label = batch.pop('label').to(nest.device)

            output = model(image)
            batch_loss = criterion(output, label)

            batch_size = image.size(0)
            total += batch_size
            loss += batch_loss * batch_size

            ground_truth.append(label.cpu().numpy())
            predictions.append((output > cfg.test.threshold).float().cpu().numpy())

    total = nest.reduce(torch.as_tensor(total, device=nest.device))
    loss = nest.reduce(loss) / total
    nest.log(f'[TEST] loss: {loss.item():.6f}')

    ground_truth = nest.gather(ground_truth, dst=0)
    predictions = nest.gather(predictions, dst=0)
    ground_truth = np.array(ground_truth).reshape(-1, test.num_classes)
    predictions = np.array(predictions).reshape(-1, test.num_classes)

    if nest.is_primary():
        test_results = utils.test_classification(
            ground_truth,
            predictions,
            test.label_names,
            6,
            folder.root / 'test.json',
            nest.log,
        )
        test_results = test_results['samples avg']
        metrics = '\n|Precision|Recall|F1-score|\n|-|-|-|\n|{precision}|{recall}|{f1}|'.format(
            precision=test_results['precision'],
            recall=test_results['recall'],
            f1=test_results['f1-score'],
        )
    else:
        metrics = None

    utils.maybe_push_to_hf_hub(
        model,
        **cfg.hub,
        model_card=dict(details=dict(metrics=metrics)),
        model_args=model_args,
    )
    nest.finish_wandb()


if __name__ == '__main__':
    main()
