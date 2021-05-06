import torch
import torch.nn as nn
from einops import rearrange
from mmcv import ConfigDict
from mmcv.cnn import build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from torch.nn.modules.utils import _pair

from mmaction.utils import trunc_normal_
from ..registry import BACKBONES


class PatchEmbed(nn.Module):
    """Image to Patch Embedding.

    Args:
        img_size (int | tuple): The size of input image.
        patch_size (int): The size of one patch
        in_channels (int): The num of input channels.
        embed_dims (int): The dimensions of embedding.
        conv_cfg (dict | None): The config dict for conv layers.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 conv_cfg=dict(type='Conv2d')):
        super().__init__()
        self.img_size = _pair(img_size)
        self.patch_size = _pair(patch_size)

        num_patches = (self.img_size[1] // self.patch_size[1]) * (
            self.img_size[0] // self.patch_size[0])
        assert num_patches * self.patch_size[0] * self.patch_size[1] == \
               self.img_size[0] * self.img_size[1], \
               'The image size H*W must be divisible by patch size'
        self.num_patches = num_patches

        # Use conv layer to embed
        self.projection = build_conv_layer(
            conv_cfg,
            in_channels,
            embed_dims,
            kernel_size=patch_size,
            stride=patch_size)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.projection(x).flatten(2).transpose(1, 2)
        return x


@BACKBONES.register_module()
class TimeSformer(nn.Module):
    supported_attention_type = [
        'divided_space_time', 'space_only', 'joint_space_time'
    ]

    def __init__(self,
                 num_frames,
                 img_size,
                 patch_size,
                 pretrained=None,
                 embed_dims=768,
                 in_channels=3,
                 drop_rate=0.,
                 transformer_layers=None,
                 attention_type='divided_space_time',
                 norm_cfg=dict(type='LN'),
                 **kwargs):
        super().__init__(**kwargs)
        assert attention_type in self.supported_attention_type, (
            f'Unsupported Attention Type {self.attention_type}!')
        self.embed_dims = embed_dims
        self.attention_type = attention_type
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=embed_dims)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(
                torch.zeros(1, num_frames, embed_dims))
            self.drop_after_time = nn.Dropout(p=drop_rate)

        if transformer_layers is None:
            if self.attention_type == 'divided_space_time':
                transformer_layers = ConfigDict(
                    dict(
                        type='TransformerLayerSequence',
                        transformerlayers=dict(
                            type='BaseTransformerLayer',
                            attn_cfgs=[
                                dict(
                                    type='DividedTemporalAttentionWithNorm',
                                    embed_dims=embed_dims,
                                    num_heads=8,
                                    num_frames=num_frames,
                                    attn_drop=0.,
                                    dropout_layer=dict(
                                        type='DropPath', drop_prob=0.1),
                                    norm_cfg=dict(type='LN')),
                                dict(
                                    type='DividedSpatialAttentionWithNorm',
                                    embed_dims=embed_dims,
                                    num_heads=8,
                                    num_frames=num_frames,
                                    attn_drop=0.,
                                    dropout_layer=dict(
                                        type='DropPath', drop_prob=0.1),
                                    norm_cfg=dict(type='LN'))
                            ],
                            ffn_cfgs=dict(
                                embed_dims=embed_dims,
                                feedforward_channels=3072,
                                num_fcs=2,
                                ffn_drop=0.,
                                act_cfg=dict(type='GELU'),
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=0.1)),
                            operation_order=('self_attn', 'self_attn', 'norm',
                                             'ffn'),
                            norm_cfg=dict(type='LN')),
                        num_layers=12))
            else:
                transformer_layers = ConfigDict(
                    dict(
                        type='TransformerLayerSequence',
                        transformerlayers=dict(
                            type='BaseTransformerLayer',
                            attn_cfgs=[
                                dict(
                                    type='MultiheadAttention',
                                    embed_dims=embed_dims,
                                    num_heads=8,
                                    dropout_layer=dict(
                                        type='DropPath', drop_prob=0.1))
                            ],
                            ffn_cfgs=dict(
                                embed_dims=embed_dims,
                                feedforward_channels=3072,
                                num_fcs=2,
                                ffn_drop=0.,
                                act_cfg=dict(type='GELU', inplace=True),
                                dropout_layer=dict(
                                    type='DropPath', drop_prob=0.1)),
                            operation_order=('norm', 'self_attn', 'norm',
                                             'ffn'),
                            norm_cfg=dict(type='LN')),
                        num_layers=12))

        self.transformer_layers = build_transformer_layer_sequence(
            transformer_layers)

        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)

    def init_weights(self):
        pass

    def forward(self, x):
        # x [batch_size * num_frames, num_patches, embed_dims]
        B = x.shape[0]
        x = self.patch_embed(x)

        # x [batch_size * num_frames, num_patches + 1, embed_dims]
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.drop_after_pos(x)

        # Time Embedding
        if self.attention_type != 'space_only':
            # x [batch_size, num_patches * num_frames + 1, embed_dims]
            x = rearrange(x[:, 1:, :], '(b t) p m -> (b p) t m', b=B)
            x = x + self.time_embed
            x = self.drop_after_time(x)
            x = rearrange(x, '(b p) t m -> b (p t) m', b=B)
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer_layers(x, None, None)

        if self.attention_type == 'space_only':
            # x [batch_size, num_patches + 1, embed_dims]
            x = x.view(-1, self.num_frames, *x.size()[-2:])
            x = torch.mean(x, 1)
        x = self.norm(x)

        return x[:, 0]
