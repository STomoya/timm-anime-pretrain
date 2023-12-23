import os

from huggingface_hub import login
from omegaconf import DictConfig
from timm.models import push_to_hf_hub
from torch.distributed.fsdp import FullyShardedDataParallel

from utils.misc import only_on_primary


def create_model_args(config: DictConfig):
    model_args = {}
    for name, value in config.items():
        # exclude 'model_name' and common model arguments not needed for inference.
        if name in ['model_name', 'drop_rate', 'drop_path_rate', 'proj_drop_rate', 'attn_drop_rate']:
            continue
        model_args[name] = value
    if model_args == {}:
        return None
    return model_args


@only_on_primary
def maybe_push_to_hf_hub(
    model,
    repo_id: str,
    commit_message: str = 'Add model',
    token='',
    revision=None,
    private: bool = False,
    create_pr: bool = False,
    model_config=None,
    model_card=None,
    safe_serialization='both',
    model_args=None,
):
    if isinstance(model, FullyShardedDataParallel):
        print('This function does not support FSDP models. Please push to hub manually. Aborting.')
        return

    token = os.environ.get('HUGGINGFACE_API_TOKEN', token)

    if token != '':
        login(token=token)
        if hasattr(model, 'dynamo_ctx'):
            model = model._orig_mod
        return push_to_hf_hub(
            model=model,
            repo_id=repo_id,
            commit_message=commit_message,
            token=token,
            revision=revision,
            private=private,
            create_pr=create_pr,
            model_config=model_config,
            model_card=model_card,
            safe_serialization=safe_serialization,
            model_args=model_args,
        )

    print('A valid HF API token was not recognized. Aborting.')
