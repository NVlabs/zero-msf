# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

# The AsymmetricMASt3R_flow class in this file extends the AsymmetricMASt3R class in https://github.com/naver/mast3r/blob/main/mast3r/model.py
# The modifications are to add scene flow estimation capabilities to the class. 
# The original file is subject to the license located at https://github.com/naver/mast3r/blob/main/LICENSE


import math
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from .dpt_head_custom import mast3r_head_factory_flow
from .utils.geometry import RGB2SH, world_to_camera_coordinates
import zmsf.utils.path_to_mast3r
import mast3r.utils.path_to_dust3r
import dust3r.utils.path_to_croco
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.misc import transpose_to_landscape
from utils.misc import get_parameter_groups


def pts_scale(
    X_cam, # Bx(2H)xWx3
    valid_mask, # Bx(2H)xW
):
    B = X_cam.shape[0]
    X_cam = X_cam.clone()
    X_cam[~valid_mask] = float('nan')
    _norm = X_cam.norm(dim=-1).view(B, -1) # Bx(2H*W)
    scale = torch.nanmean(_norm, dim=1).view(-1)
    return scale


inf = float('inf')


class AsymmetricMASt3R_flow(AsymmetricCroCo3DStereo):
    def __init__(self, 
        flow_mode=None,
        infer_flow:Optional[bool]=False,
        desc_mode=('norm'), 
        two_confs=False, 
        desc_conf_mode=None, 
        use_gt_pts3d=False,  # added for compatibility with full code
        max_sh_degree=1,  # added for compatibility with full code
        custom_profiler=False,  # added for compatibility with full code
        **kwargs):
        self.flow_mode = flow_mode
        self.infer_flow = infer_flow
        self.desc_mode = desc_mode
        self.two_confs = two_confs
        self.desc_conf_mode = desc_conf_mode
        self.use_gt_pts3d = use_gt_pts3d
        self.max_sh_degree = max_sh_degree
        self.sh_num = (self.max_sh_degree + 1) ** 2 - 1 # how many sh coefficients in total
        
        self.custom_profiler = custom_profiler
        if self.custom_profiler:
            self.setup_profiler()
        
        super().__init__(**kwargs)
    
    def training_setup(self, lr, weight_decay):
        param_groups = get_parameter_groups(self, weight_decay)
        self.optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=(0.9, 0.95))
    
    def adjust_learning_rate(self, epoch, warmup_epochs, lr, min_lr, epochs):
        """Decay the learning rate with half-cycle cosine after warmup"""
        
        if epoch < warmup_epochs:
            lr = lr * epoch / warmup_epochs 
        else:
            lr = min_lr + (lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - warmup_epochs) / (epochs - warmup_epochs)))
        return lr

    def setup_profiler(self) -> None:
        assert self.custom_profiler
        self.timing_events = {
            'forward_start': torch.cuda.Event(enable_timing=True),
            'forward_encode': torch.cuda.Event(enable_timing=True),
            'forward_decode': torch.cuda.Event(enable_timing=True),
            'forward_heads': torch.cuda.Event(enable_timing=True),
            'forward_gt3d': torch.cuda.Event(enable_timing=True),
            'forward_gaussians': torch.cuda.Event(enable_timing=True),
            'forward_offset': torch.cuda.Event(enable_timing=True),
            'forward_end': torch.cuda.Event(enable_timing=True),
        }

    def set_downstream_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode, patch_size, img_size, **kw):
        assert img_size[0] % patch_size == 0 and img_size[
            1] % patch_size == 0, f'{img_size=} must be multiple of {patch_size=}'
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        if self.desc_conf_mode is None:
            self.desc_conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = mast3r_head_factory_flow(head_type, output_mode, self, has_conf=bool(conf_mode))
        self.downstream_head2 = mast3r_head_factory_flow(head_type, output_mode, self, has_conf=bool(conf_mode))
        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def forward(self, input_dict: Dict) -> Dict:
        if self.custom_profiler:
            self.timing_events['forward_start'].record()


        ######################### Dust3r predict point cloud ##########################
        # encode the two images --> B,S,D
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self._encode_symmetrized(input_dict["view1"], input_dict["view2"])


        if self.custom_profiler:
            self.timing_events['forward_encode'].record()


        # combine all ref images into object-centric representation
        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)

        if self.custom_profiler:
            self.timing_events['forward_decode'].record()


        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

        if self.custom_profiler:
            self.timing_events['forward_heads'].record()

        #################### (optional) get real flow by subtraction ###############
        if self.infer_flow:
            res1["flow"] -= res1["pts3d"]
            res2["flow"] -= res2["pts3d"]

        # potentially align scale given groundtruth
        if "align_scale" in input_dict and input_dict["align_scale"]:
            # compute scaler (should not related to gradient flow)
            with torch.no_grad():
                gt_pts1 = world_to_camera_coordinates(input_dict["view1"]["pts3d"], input_dict["view1"]["camera_pose"]) 
                gt_pts2 = world_to_camera_coordinates(input_dict["view2"]["pts3d"], input_dict["view1"]["camera_pose"]) 
                valid1 = input_dict["view1"]['valid_mask'].bool()
                valid2 = input_dict["view2"]['valid_mask'].bool()        
                
                # get scale first
                gt_scale = pts_scale(
                    torch.cat([gt_pts1, gt_pts2], dim=1), 
                    torch.cat([valid1, valid2], dim=1)
                )
                pred_scale = pts_scale(
                    torch.cat([res1["pts3d"], res2["pts3d"]], dim=1),
                    torch.cat([valid1, valid2], dim=1)
                )
                scaler = gt_scale / (pred_scale+1e-6)
                scaler = scaler[:, None, None, None]
            res1["pts3d"] *= scaler
            res2["pts3d"] *= scaler
            if input_dict["decompose"]:
                res1["flow"][..., 2:] *= scaler
                res2["flow"][..., 2:] *= scaler
            else:
                res1["flow"] *= scaler
                res2["flow"] *= scaler



        ######################### (optional) Unproj RGB-D ##########################        
        if self.use_gt_pts3d:
            # proj both to camera coordinate
            res1["pts3d"] = world_to_camera_coordinates(input_dict["view1"]["pts3d"], input_dict["view1"]["camera_pose"]).float()
            # relative to first camera instead of second camera
            res2["pts3d"] = world_to_camera_coordinates(input_dict["view2"]["pts3d"], input_dict["view1"]["camera_pose"]).float() 

        if self.custom_profiler:
            self.timing_events['forward_gt3d'].record()

        if "camera_intrinsics" in input_dict["view1"]:
            ######################### Load other Gaussian parameters origin ############       
            B, H, W, _ = res1["pts3d"].shape
            res1["opacity"] = torch.ones(B, H, W, 1, device=res1["pts3d"].device) * 0.5     
            res1["rotation"] = torch.ones(B, H, W, 4, device=res1["pts3d"].device)
            res1["rotation"][..., 1:] *= 0.
            # depending on coordinate system!!!
            res1["scaling"] = res1["pts3d"][..., -1].detach() / input_dict["view1"]["camera_intrinsics"][0, 0, 0]     
            res1["scaling"] = res1["scaling"][..., None].expand(B, H, W, 3)

            res1["features_dc"] = input_dict["view1"]["img"].permute(0, 2, 3, 1)[..., None, :] # BxHxWx1x3
            res1["features_dc"] = res1["features_dc"]/2. + .5
            res1["features_dc"] = RGB2SH(res1["features_dc"])
            
            res1["features_rest"] = torch.zeros(B, H, W, self.sh_num, 3, device=res1["pts3d"].device) # BxHxWx?x3
            
            res2["opacity"] = torch.ones(B, H, W, 1, device=res2["pts3d"].device) * 0.5     
            res2["rotation"] = torch.ones(B, H, W, 4, device=res2["pts3d"].device)
            res2["rotation"][..., 1:] *= 0.
            # for now referenced to left camera, thus use left camera's focal length
            # depending on coordinate system!!!
            res2["scaling"] = res2["pts3d"][..., -1].detach() / input_dict["view1"]["camera_intrinsics"][0, 0, 0]
            res2["scaling"] = res2["scaling"][..., None].expand(B, H, W, 3)

            res2["features_dc"] = input_dict["view2"]["img"].permute(0, 2, 3, 1)[..., None, :] # BxHxWx1x3
            res2["features_dc"] = res2["features_dc"]/2. + .5
            res2["features_dc"] = RGB2SH(res2["features_dc"])

            res2["features_rest"] = torch.zeros(B, H, W, self.sh_num, 3, device=res2["pts3d"].device) # BxHxWx?x3
        else:
            res1["opacity"] = None
            res1["rotation"] = None
            res1["scaling"] = None
            res1["features_dc"] = None
            res1["features_rest"] = None
            res2["opacity"] = None
            res2["rotation"] = None
            res2["scaling"] = None
            res2["features_dc"] = None
            res2["features_rest"] = None

        if self.custom_profiler:
            self.timing_events['forward_gaussians'].record()

        ######################### Load Gaussian parameters offsets origin ############  
        
        
        if self.custom_profiler:
            self.timing_events['forward_offset'].record()


        ######################### Put offset computations here ##################


        ######################### Prepare return Dict ###########################

        # all below assume after activation already!
        merged_res = {
            ## first timestep, a part of visible gaussian 
            "xyz_1": res1["pts3d"],
            "conf_1": res1["conf"], # before sigmoid
            "flow_1": res1["flow"],
            "conf_flow_1": res1["conf_flow"],
            "scaling_1": res1["scaling"],
            "opacity_1": res1["opacity"],
            "rotation_1": res1["rotation"],
            "features_dc_1": res1["features_dc"],
            "features_rest_1": res1["features_rest"],
            ## second timestep, a part of visible gaussian
            "xyz_2": res2["pts3d"], 
            "conf_2": res2["conf"], # before sigmoid
            "flow_2": res2["flow"],
            "conf_flow_2": res2["conf_flow"],
            "scaling_2": res2["scaling"],
            "opacity_2": res2["opacity"],
            "rotation_2": res2["rotation"],
            "features_dc_2": res2["features_dc"],
            "features_rest_2": res2["features_rest"],
            "max_sh_degree": self.max_sh_degree
        }

        
        # Step 3: Delete the original dictionaries
        del res1
        del res2

        if self.custom_profiler:
            self.timing_events['forward_end'].record()
        
        return merged_res
