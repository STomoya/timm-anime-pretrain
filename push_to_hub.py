import argparse

import timm
import torch
from omegaconf import OmegaConf
from storch.io import load_json
from storch.path import Path

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Path to the config file of the model to upload.')
    args = parser.parse_args()

    # Path to config file.
    config_path = args.config

    # files
    project_dir = Path(config_path).dirname()
    test_result_path = project_dir / 'test.json'
    best_model_path = project_dir / 'loss.torch'

    if not test_result_path.exists():
        raise Exception('The training seems to be incomplete. Aborting.')

    # load config
    config = OmegaConf.load(config_path)
    cfg = config.config

    # setup input size
    # utils.set_input_size(cfg.model.model_name, cfg.data)

    # create dataset for determining number of classes.
    train, _, _ = utils.build_dataset(cfg.data)

    # build model
    cfg.model.pop('pretrained', None)
    tag = cfg.hub.repo_id.split('.')[-1]
    model = timm.create_model(
        **cfg.model,
        num_classes=train.num_classes,
        pretrained_cfg_overlay=dict(
            url='',
            custom_load=True,
            num_classes=train.num_classes,
            mean=utils.DATASET_MEAN,
            std=utils.DATASET_STD,
            hf_hub_id=cfg.hub.repo_id,
            tag=tag,
        ),
    )
    # collect args for building model.
    model_args = utils.create_model_args(cfg.model)

    # load state_dict
    state_dict = torch.load(best_model_path, map_location='cpu')['state_dict']
    model.load_state_dict(state_dict)

    # load test results
    test_results = load_json(test_result_path)
    test_results = test_results['samples avg']
    metrics = '\n|Precision|Recall|F1-score|\n|-|-|-|\n|{precision}|{recall}|{f1}|'.format(
        precision=test_results['precision'],
        recall=test_results['recall'],
        f1=test_results['f1-score'],
    )

    # push to hub
    utils.maybe_push_to_hf_hub(
        model,
        **cfg.hub,
        model_card=dict(details=dict(metrics=metrics)),
        model_args=model_args,
    )


if __name__ == '__main__':
    main()
