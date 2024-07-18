import glob
import os

import torch.nn as nn
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from storch.dataset import imagepaths, make_transform_from_config
from storch.hydra_utils import to_object


def build_dataset(config):
    csv_files = glob.glob(os.path.join(config.dataset_root, '*.csv'))
    image_paths = []
    for csv_file in csv_files:
        with open(csv_file, 'r') as fp:
            samples = fp.read().strip().splitlines()
        for sample in samples:
            image_path = sample.split(',')[0]
            image_paths.append(image_path)

    transforms = make_transform_from_config(to_object(config.transforms.train))
    dataset = imagepaths(paths=image_paths, transforms=transforms)
    return dataset


class MAE(nn.Module):
    def __init__(
        self,
        vit,
        mask_ratio: float = 0.75,
        decoder_dim: float = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int | None = None,
        decoder_mlp_ratio: float = 4.0,
        decoder_proj_drop_rate: float = 0.0,
        decoder_attn_drop_rate: float = 0.0,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = vit.patch_embed.patch_size[0]

        if decoder_dim <= 1.0:
            # MAE decoders are usually smaller than the encoder
            decoder_dim = vit.embed_dim * decoder_dim
        if decoder_num_heads is None:
            # If not specified default to num_heads of the encoder.
            decoder_num_heads = vit.blocks[0].attn.num_heads

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=int(decoder_dim),
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=decoder_mlp_ratio,
            proj_drop_rate=decoder_proj_drop_rate,
            attn_drop_rate=decoder_attn_drop_rate,
        )

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(self.decoder.mask_token, (batch_size, self.sequence_length))
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        return x_pred, target
