from storch.hydra_utils import get_hydra_config, save_hydra_config


def main():
    config = get_hydra_config('config', 'config.yaml', resolve=False)
    config.config.data.image_size=384
    save_hydra_config(config, './config.yaml')


if __name__=='__main__':
    main()
