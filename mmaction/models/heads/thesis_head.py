import copy
import math

import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import constant_init

from mmaction.models.common import LFB

try:
    from mmdet.models.builder import SHARED_HEADS as MMDET_SHARED_HEADS
    mmdet_imported = True
except (ImportError, ModuleNotFoundError):
    mmdet_imported = False


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of
    residual blocks).

    We follow the implementation
    https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
    """

    def __init__(self, drop_prob=0.1):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class FBOThesis(nn.Module):
    """FBO Thesis."""

    def __init__(
            self,
            st_feat_channels,
            lt_feat_channels,
            latent_channels,
            window_size=60,
            time_embedding_style='fixed',  # 'learnable'
            st_feat_dropout_ratio=0.2,
            lt_feat_dropout_ratio=0.2):
        super().__init__()
        self.st_feat_channels = st_feat_channels
        self.lt_feat_channels = lt_feat_channels
        self.latent_channels = latent_channels
        self.window_size = window_size
        self.time_embedding_style = time_embedding_style
        self.st_feat_dropout_ratio = st_feat_dropout_ratio
        self.lt_feat_dropout_ratio = lt_feat_dropout_ratio

        self.st_feat_proj = nn.Linear(st_feat_channels, latent_channels)
        self.lt_feat_proj = nn.Linear(lt_feat_channels, latent_channels)

        if self.st_feat_dropout_ratio > 0:
            self.st_feat_dropout = nn.Dropout(self.st_feat_dropout_ratio)

        if self.lt_feat_dropout_ratio > 0:
            self.lt_feat_dropout = nn.Dropout(self.lt_feat_dropout_ratio)

        if self.time_embedding_style == 'learnable':
            self.time_embed = nn.Parameter(
                torch.zeros(1, window_size, latent_channels))
        else:
            self.time_embed = nn.Parameter(
                torch.zeros(1, window_size, latent_channels),
                requires_grad=False)
            # following attention is all your need
            for pos in range(window_size):
                for i in range(0, latent_channels, 2):
                    self.time_embed[0, pos, i] = math.sin(
                        pos / (10000**((2 * i) / latent_channels)))
                    self.time_embed[0, pos, i + 1] = math.cos(
                        pos / (10000**((2 * i + 2) / latent_channels)))

        self.temporal_norm = nn.LayerNorm(latent_channels)
        self.temporal_attn = nn.MultiheadAttention(
            latent_channels, num_heads=8)
        self.drop_path = DropPath()
        self.temporal_fc = nn.Linear(latent_channels, latent_channels)

        self.st_feat_norm = nn.LayerNorm(latent_channels)
        self.lt_feat_norm = nn.LayerNorm(latent_channels)
        self.spatial_attn = nn.MultiheadAttention(latent_channels, num_heads=8)

    def init_weights(self, pretrained=None):
        # zero init temporal_fc
        constant_init(self.temporal_fc, val=0, bias=0)

    def forward(self, st_feat, lt_feat):
        # st_feat list of each video's roi_featurs
        # lt_feat [B, window_size * max_num_feat_per_step, lfb_channels]
        st_feat = list(map(self.st_feat_proj, st_feat))
        if self.st_feat_dropout_ratio > 0:
            st_feat = list(map(self.st_feat_dropout, st_feat))

        lt_feat = self.lt_feat_proj(lt_feat)
        if self.lt_feat_dropout_ratio > 0:
            lt_feat = self.lt_feat_dropout(lt_feat)

        B, T = lt_feat.size(0), self.window_size

        # temporal attention
        res_lt_feat = rearrange(lt_feat, 'b (t k) c -> (b k) t c', t=T)
        res_lt_feat = res_lt_feat + self.time_embed

        res_lt_feat = self.temporal_norm(res_lt_feat).permute(1, 0, 2)
        res_lt_feat = self.temporal_attn(res_lt_feat, res_lt_feat,
                                         res_lt_feat)[0].permute(1, 0, 2)
        res_lt_feat = self.drop_path(res_lt_feat)

        res_lt_feat = rearrange(res_lt_feat, '(b k) t c -> b (t k) c', b=B)
        res_lt_feat = self.temporal_fc(res_lt_feat)

        lt_feat += res_lt_feat

        # TODO: add spatial position embedding

        # spatial attention
        def spatial_attention(_st_feat, _lt_feat):
            # [num_rois, 1, 512] [window_size * num_rois, 1, 512]
            identity = _st_feat
            _st_feat = self.st_feat_norm(_st_feat).unsqueeze(1)
            _lt_feat = self.lt_feat_norm(_lt_feat).unsqueeze(1)
            _fbo_feat = self.spatial_attn(_st_feat, _lt_feat, _lt_feat)[0]
            _fbo_feat = self.drop_path(_fbo_feat.squeeze(1))
            _fbo_feat += identity
            return _fbo_feat

        fbo_feat = list(map(spatial_attention, st_feat, lt_feat))

        return fbo_feat


class ThesisHead(nn.Module):
    """Thesis Head.

    Add feature bank operator for the spatiotemporal detection model to fuse
    short-term features and long-term features.

    Args:
        lfb_cfg (Dict): The config dict for LFB which is used to sample
            long-term features.
        fbo_cfg (Dict): The config dict for feature bank operator (FBO). The
            type of fbo is also in the config dict and supported fbo type is
            `fbo_dict`.
        temporal_pool_type (str): The temporal pool type. Choices are 'avg' or
            'max'. Default: 'avg'.
        spatial_pool_type (str): The spatial pool type. Choices are 'avg' or
            'max'. Default: 'max'.
    """

    fbo_dict = {'thesis': FBOThesis}

    def __init__(self,
                 lfb_cfg,
                 fbo_cfg,
                 pretrained=None,
                 with_local=True,
                 temporal_pool_type='avg',
                 spatial_pool_type='max'):
        super().__init__()
        fbo_type = fbo_cfg.pop('type', 'thesis')
        assert fbo_type in ThesisHead.fbo_dict
        assert temporal_pool_type in ['max', 'avg']
        assert spatial_pool_type in ['max', 'avg']

        self.lfb_cfg = copy.deepcopy(lfb_cfg)
        self.fbo_cfg = copy.deepcopy(fbo_cfg)

        self.lfb = LFB(**self.lfb_cfg)
        self.fbo = self.fbo_dict[fbo_type](**self.fbo_cfg)

        self.with_local = with_local

        # Pool by default
        if temporal_pool_type == 'avg':
            self.temporal_pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            self.temporal_pool = nn.AdaptiveMaxPool3d((1, None, None))
        if spatial_pool_type == 'avg':
            self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        else:
            self.spatial_pool = nn.AdaptiveMaxPool3d((None, 1, 1))

    def init_weights(self, pretrained=None):
        """Initialize the weights in the module.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        self.fbo.init_weights(pretrained=pretrained)

    def sample_lfb(self, img_metas):
        """Sample long-term features for each ROI feature."""
        lt_feat_list = []
        for img_meta in img_metas:
            lt_feat_list.append(self.lfb[img_meta['img_key']])

        # [B, window_size, max_num_feat_per_step, lfb_channels]
        lt_feat = torch.stack(lt_feat_list, dim=0)
        return lt_feat.view(lt_feat.size(0), -1, lt_feat.size(-1))

    def get_st_feat_by_epoch(self, st_feat, rois, img_metas):
        inds = rois[:, 0].type(torch.int64)
        B = len(img_metas)

        st_feat_by_epoch = [st_feat[inds == batch_id] for batch_id in range(B)]

        return st_feat_by_epoch

    def forward(self, x, rois, img_metas):
        # [N, C]
        N, C, _, _, _ = x.shape
        st_feat = self.spatial_pool(self.temporal_pool(x))
        identity = st_feat = st_feat.reshape(N, C)

        # [B, window_size * max_num_feat_per_step, lfb_channels]
        lt_feat = self.sample_lfb(img_metas).to(x.device)

        # list of each video's roi_featurs
        st_feat = self.get_st_feat_by_epoch(st_feat, rois, img_metas)

        # [B, C]
        fbo_feat = self.fbo(st_feat, lt_feat)

        # organize fbo_feat
        inds = rois[:, 0].type(torch.int64)
        global_fbo_feats = torch.stack(
            list(map(lambda x: torch.mean(x, dim=0), fbo_feat)), dim=0)
        global_fbo_feat = global_fbo_feats[inds]
        # [N, C + 512]
        out = torch.cat([identity, global_fbo_feat], dim=1)

        if self.with_local:
            local_fbo_feat = torch.zeros(N, global_fbo_feat.size(-1))
            for idx in range(N):
                batch_id = inds[idx]
                local_fbo_feat[idx] = fbo_feat[batch_id][torch.sum(
                    inds[:idx] == batch_id)]
            # [N, C + 512 + 512]
            out = torch.cat([identity, local_fbo_feat], dim=1)

        return out.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)


if mmdet_imported:
    MMDET_SHARED_HEADS.register_module()(ThesisHead)
