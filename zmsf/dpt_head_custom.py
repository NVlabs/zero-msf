# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

# This file extends parts of https://github.com/naver/mast3r/blob/main/mast3r/catmlp_dpt_head.py
# The modifications are to add scene flow estimation capabilities to the functions. 
# The original file is subject to the license located at https://github.com/naver/mast3r/blob/main/LICENSE


import torch
import torch.nn.functional as F

import zmsf.utils.path_to_mast3r
import mast3r.utils.path_to_dust3r
from dust3r.heads.postprocess import reg_dense_depth, reg_dense_conf 
from dust3r.heads.dpt_head import PixelwiseTaskWithDPT, DPTOutputAdapter_fix


def reg_dense_flow(xyz, mode):
    """
    extract 3D flow from prediction head output
    """
    assert mode is not None
    return reg_dense_depth(xyz, mode)
    
    vmin = -float('inf')
    vmax = float('inf')
    return xyz  # [-inf, +inf]


def postprocess(out, depth_mode, conf_mode, flow_mode, desc_dim=None, desc_mode='norm', two_confs=False, desc_conf_mode=None):
    if desc_conf_mode is None:
        desc_conf_mode = conf_mode
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,D
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))
    if conf_mode is not None:
        res['conf'] = reg_dense_conf(fmap[..., 3], mode=conf_mode)
        res['flow'] = reg_dense_flow(fmap[..., 4:7], mode=flow_mode)
        if two_confs:
            res['conf_flow'] = reg_dense_conf(fmap[..., 7], mode=conf_mode)
        else:
            res['conf_flow'] = res['conf'].clone()
    else:
        res['flow'] = reg_dense_flow(fmap[..., 3:6], mode=flow_mode)
    
    return res


class Cat_MLP_LocalFeatures_DPT_Pts3d_flow(PixelwiseTaskWithDPT):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """
    def __init__(self, net, has_conf=False, local_feat_dim=16, hidden_dim_factor=4., hooks_idx=None, dim_tokens=None,
                 num_channels=1, postprocess=None, feature_dim=256, last_dim=32, depth_mode=None, conf_mode=None, flow_mode=None, head_type="regression", **kwargs):  
        # Pt3d heads: self.dpt
        super().__init__(
            num_channels=num_channels, # not kwargs
            feature_dim=feature_dim, # kwargs
            last_dim=last_dim, # kwargs
            hooks_idx=hooks_idx, # not kwargs
            dim_tokens=dim_tokens, # not kwargs
            depth_mode=depth_mode, # not kwargs
            postprocess=postprocess, # not kwargs
            conf_mode=conf_mode, # not kwargs
            head_type=head_type # kwargs
            )
        self.flow_mode = flow_mode

        # Flow Heads: self.dpt_flow
        dpt_args = dict(output_width_ratio=1,
                        num_channels=num_channels,
                        #kwargs
                        feature_dim=feature_dim, 
                        last_dim=last_dim, 
                        head_type=head_type)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt_flow = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt_flow.init(**dpt_init_args)
        
        # Local Feat Heads
        self.local_feat_dim = local_feat_dim

        patch_size = net.patch_embed.patch_size
        if isinstance(patch_size, tuple):
            assert len(patch_size) == 2 and isinstance(patch_size[0], int) and isinstance(
                patch_size[1], int), "What is your patchsize format? Expected a single int or a tuple of two ints."
            assert patch_size[0] == patch_size[1], "Error, non square patches not managed"
            patch_size = patch_size[0]
        self.patch_size = patch_size

        self.desc_mode = net.desc_mode
        self.has_conf = has_conf
        self.two_confs = net.two_confs  # independent confs for 3D regr and descs
        self.desc_conf_mode = net.desc_conf_mode
        idim = net.enc_embed_dim + net.dec_embed_dim
        
    def forward(self, decout, img_shape):
        # pass through the heads
        
        # pt3d
        pts3d = self.dpt(decout, image_size=(img_shape[0], img_shape[1]))

        # flow
        flow = self.dpt_flow(decout, image_size=(img_shape[0], img_shape[1]))

        
        out = torch.cat([pts3d, flow], dim=1)
        
        if self.postprocess:
            out = self.postprocess(out,
                                   depth_mode=self.depth_mode,
                                   conf_mode=self.conf_mode,
                                   flow_mode=self.flow_mode,
                                   desc_dim=self.local_feat_dim,
                                   desc_mode=self.desc_mode,
                                   two_confs=self.two_confs,
                                   desc_conf_mode=self.desc_conf_mode)
        return out


def mast3r_head_factory_flow(head_type, output_mode, net, has_conf=False):
    """" build a prediction head for the decoder 
    """
    if head_type == 'catmlp+dpt' and output_mode.startswith('pts3d+desc'):
        local_feat_dim = int(output_mode[10:])
        assert net.dec_depth > 9
        l2 = net.dec_depth
        feature_dim = 256
        last_dim = feature_dim // 2
        out_nchan = 3
        ed = net.enc_embed_dim
        dd = net.dec_embed_dim
        return Cat_MLP_LocalFeatures_DPT_Pts3d_flow(net, local_feat_dim=local_feat_dim, has_conf=has_conf,
                                               num_channels=out_nchan + has_conf,
                                               feature_dim=feature_dim,
                                               last_dim=last_dim,
                                               hooks_idx=[0, l2 * 2 // 4, l2 * 3 // 4, l2],
                                               dim_tokens=[ed, dd, dd, dd],
                                               postprocess=postprocess,
                                               depth_mode=net.depth_mode,
                                               conf_mode=net.conf_mode,
                                               flow_mode=net.flow_mode,
                                               head_type='regression')
    else:
        raise NotImplementedError(
            f"unexpected {head_type=} and {output_mode=}")
