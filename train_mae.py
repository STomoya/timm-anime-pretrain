"""MAE."""

import gc

import timm
import torch
import torch.nn as nn
from PIL import ImageFile
from storch.hydra_utils import to_object
from storch.nest import NeST

import utils
from utils.ssl.mae import MAE, build_dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    config, folder, save_config = utils.init_project(default_config_file='config.mae.yaml')

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

    utils.set_input_size(cfg.model.vit.model_name, cfg.data)

    # build dataset
    dcfg = cfg.data
    train = build_dataset(dcfg)
    trainloader = nest.build_dataloader(
        is_train=True,
        dataset=train,
        **dcfg.loader.train,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )

    # build model
    vit = timm.create_model(
        **cfg.model.vit,
        pretrained_cfg_overlay=dict(
            url='',
            custom_load=True,
            num_classes=0,
            mean=utils.DATASET_MEAN,
            std=utils.DATASET_STD,
            hf_hub_id=cfg.hub.repo_id,
            tag='st_mae_sb1k',
        ),
    )
    model_args = utils.create_model_args(cfg.model.vit)
    assert isinstance(vit, timm.models.VisionTransformer)
    assert vit.num_prefix_tokens in {0, 1}, 'Currently does not support register tokens.'
    model = nest.build_model(builder=MAE, vit=vit, **cfg.model.decoder)

    # build optimizer
    ## batch size per optimization step.
    batch_size = nest.world_size * nest._grad_accum_steps * dcfg.loader.train.batch_size
    cfg.optimizer.lr = cfg.train.base_lr * batch_size / 256
    optimizer = nest.build_optimizer(**to_object(cfg.optimizer), parameters=model.parameters())

    _dummy_input = torch.randn(3, 3, 224, 224, device=nest.device)
    _dummy_output = model(_dummy_input)
    _dummy_output[0].sum().backward()
    optimizer.step()
    optimizer.zero_grad()

    # scheduler
    scheduler = nest.build_scheduler(optimizer, **cfg.scheduler)

    # criterion
    criterion = nn.MSELoss()

    # initialize training
    wandb_args = (
        dict(
            wandb_project='timm-anime-pretrain-mae',
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
        to_log=[config],
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

            reconstructed, target = model(image)
            batch_loss = criterion(reconstructed, target)

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

        nest.update_best_model(loss)
        nest.save()

        gc.collect()

    nest.load_best_model()
    model.eval()

    if cfg.hub.repo_id is not None:
        utils.maybe_push_to_hf_hub(
            vit,
            **cfg.hub,
            model_card=dict(details=dict(MSE=f'{loss.item()}')),
            model_args=model_args,
        )
    nest.finish_wandb()


if __name__ == '__main__':
    main()
