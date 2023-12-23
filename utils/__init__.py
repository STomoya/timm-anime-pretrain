from utils.constants import *
from utils.data import MultiLabelCSV
from utils.hub import create_model_args, maybe_push_to_hf_hub
from utils.metrics import test_classification
from utils.misc import is_primary, only_on_primary
from utils.randaugment import RandAugment


def init_project(
    config_file: str = 'config.yaml', config_dir: str = 'config', default_config_file: str = 'config.yaml'
):
    import sys

    import storch
    from omegaconf import OmegaConf
    from storch.hydra_utils import get_hydra_config, save_hydra_config
    from storch.path import Folder, Path

    cmdargs = sys.argv[1:]

    # for resuming:
    # $ python3 train.py　./path/to/config.yaml
    if len(cmdargs) == 1 and cmdargs[0].endswith(config_file):
        config_path = cmdargs[0]
        config = OmegaConf.load(config_path)
        folder = Folder(Path(config_path).dirname())

        def save_config():
            return

    # for a new run.
    else:
        # load config
        config = get_hydra_config(config_dir, default_config_file)

        # create project folder
        folder = config.run.folder
        name = config.run.name
        tag = config.run.tag
        if tag == 'date':
            tag = storch.get_now_string()
        folder = Folder(Path(folder) / f'{name}.{tag}')
        if is_primary():
            folder.mkdir()

        def save_config():
            """function to save config. call after changing the config according to the env."""
            save_hydra_config(config, folder.root / config_file)

        save_config = only_on_primary(save_config)

    return config, folder, save_config


def set_input_size(model_name, data_config):
    import timm

    pretrained_cfg = timm.models.get_pretrained_cfg(model_name)
    data_config.image_size = pretrained_cfg.input_size[-1]
    test_input_size = pretrained_cfg.test_input_size
    data_config.test_image_size = test_input_size[-1] if test_input_size is not None else pretrained_cfg.input_size[-1]


def build_dataset(config):
    import os

    from storch.dataset import make_transform_from_config
    from storch.hydra_utils import to_object
    from torchvision.transforms import RandAugment as TVRandAugment

    train_transform = make_transform_from_config(to_object(config.transforms.train))

    # replace RandAugment if any
    for i in range(len(train_transform.transforms)):
        augment_fn = train_transform.transforms[i]
        if isinstance(augment_fn, TVRandAugment):
            print('found pytorch RandAugment')
            train_transform.transforms[i] = RandAugment(
                num_ops=augment_fn.num_ops,
                magnitude=augment_fn.magnitude,
                num_magnitude_bins=augment_fn.num_magnitude_bins,
                interpolation=augment_fn.interpolation,
                fill=augment_fn.fill,
            )

    test_transform = make_transform_from_config(to_object(config.transforms.test))

    train_dataset = MultiLabelCSV(
        os.path.join(config.dataset_root, 'train'), config.label_mapping_csv, train_transform, config.label_is_indexed
    )
    validation_dataset = MultiLabelCSV(
        os.path.join(config.dataset_root, 'validation'),
        config.label_mapping_csv,
        test_transform,
        config.label_is_indexed,
    )
    test_dataset = MultiLabelCSV(
        os.path.join(config.dataset_root, 'test'), config.label_mapping_csv, test_transform, config.label_is_indexed
    )

    return train_dataset, validation_dataset, test_dataset
