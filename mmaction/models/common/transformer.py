import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.registry import ATTENTION, FEEDFORWARD_NETWORK
from mmcv.cnn.bricks.transformer import FFN, build_dropout
from mmcv.runner.base_module import BaseModule


@ATTENTION.register_module()
class DividedTemporalAttentionWithNorm(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]
        self.temporal_fc = nn.Linear(self.embed_dims, self.embed_dims)

        nn.init.constant_(self.temporal_fc.weight, 0)
        nn.init.constant_(self.temporal_fc.bias, 0)

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        assert residual is None, (
            'Cannot apply pre-norm with DividedTemporalAttentionWithNorm')

        init_cls_token = query[:, 0, :].unsqueeze(1)
        identity = query_t = query[:, 1:, :]

        # query_t [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_t.size()
        p, t = pt // self.num_frames, self.num_frames

        # res_temporal [batch_size * num_patches, num_frames, embed_dims]
        query_t = self.norm(query_t.reshape(b * p, t, m)).permute(1, 0, 2)
        res_temporal = self.attn(query_t, query_t, query_t)[0].permute(1, 0, 2)
        res_temporal = self.dropout_layer(res_temporal.contiguous())
        res_temporal = self.temporal_fc(res_temporal)

        # res_temporal [batch_size, num_patches * num_frames, embed_dims]
        res_temporal = res_temporal.reshape(b, p * t, m)

        # ret_value [batch_size, num_patches * num_frames + 1, embed_dims]
        new_query_t = identity + res_temporal
        new_query = torch.cat((init_cls_token, new_query_t), 1)
        return new_query


@ATTENTION.register_module()
class DividedSpatialAttentionWithNorm(BaseModule):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 num_frames,
                 attn_drop=0.,
                 dropout_layer=dict(type='DropPath', drop_prob=0.1),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

    def forward(self, query, key=None, value=None, residual=None, **kwargs):
        assert residual is None, (
            'Cannot apply pre-norm with DividedTemporalAttentionWithNorm')

        identity = query
        init_cls_token = query[:, 0, :].unsqueeze(1)
        query_s = query[:, 1:, :]

        # query_s [batch_size, num_patches * num_frames, embed_dims]
        b, pt, m = query_s.size()
        p, t = pt // self.num_frames, self.num_frames

        # cls_token [batch_size * num_frames, 1, embed_dims]
        cls_token = init_cls_token.repeat(1, t, 1).reshape(b * t,
                                                           m).unsqueeze(1)

        # query_s [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = rearrange(query_s, 'b (p t) m -> (b t) p m', p=p, t=t)
        query_s = torch.cat((cls_token, query_s), 1)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        query_s = self.norm(query_s).permute(1, 0, 2)
        res_spatial = self.attn(query_s, query_s, query_s)[0].permute(1, 0, 2)
        res_spatial = self.dropout_layer(res_spatial.contiguous())

        # cls_token [batch_size, 1, embed_dims]
        cls_token = res_spatial[:, 0, :].reshape(b, t, m)
        cls_token = torch.mean(cls_token, 1, True)

        # res_spatial [batch_size * num_frames, num_patches + 1, embed_dims]
        res_spatial = rearrange(
            res_spatial[:, 1:, :], '(b t) p m -> b (p t) m', p=p, t=t)
        res_spatial = torch.cat((cls_token, res_spatial), 1)

        new_query = identity + res_spatial
        return new_query


@FEEDFORWARD_NETWORK.register_module()
class FFNWithNorm(FFN):

    def __init__(self, *args, norm_cfg=dict(type='LN'), **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = build_norm_layer(norm_cfg, self.embed_dims)[1]

    def forward(self, x, residual=None):
        assert residual is None, ('Cannot apply pre-norm with FFNWithNorm')
        return super().forward(self.norm(x), x)
