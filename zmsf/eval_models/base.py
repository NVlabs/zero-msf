# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/


from typing import List, Dict
import random
import math

import numpy as np
import torch

import zmsf.utils.path_to_mast3r
import mast3r.utils.path_to_dust3r
from dust3r.utils.geometry import xy_grid, find_reciprocal_matches
from dust3r_visloc.localization import run_pnp


random.seed(0)


def get_pose(
    output_dict: Dict,
    K,
    pnp_max_points=100000,
    reprojection_error=5.,
    reprojection_error_diag_ratio=None,
    pnp_mode="cv2",
):
    B, H, W, _ = output_dict["xyz_1"].shape
    PQs = output_dict["xyz_1"].detach().view(B, -1, 3).cpu().numpy()
    PMs = output_dict["xyz_2"].detach().view(B, -1, 3).cpu().numpy()

    cam2toworlds = []
    for i in range(B):
        PQ, PM = PQs[i], PMs[i] #Nx3, Nx3
        reciprocal_in_PM, nnM_in_PQ, num_matches = find_reciprocal_matches(PQ, PM)
        
        # don't need to rescale coordinate as in dust3r's visloc.py
        matches_im1 = xy_grid(W=W, H=H).reshape((-1, 2))[reciprocal_in_PM]
        matches_im0 = xy_grid(W=W, H=H).reshape((-1, 2))[nnM_in_PQ][reciprocal_in_PM]

        query_pts2d = matches_im0 + 0.
        query_pts3d = PM.reshape((H, W, 3))[matches_im1[:, 1], matches_im1[:, 0]]

        if len(query_pts2d) > pnp_max_points:
            idxs = random.sample(range(len(query_pts2d)), pnp_max_points)
            query_pts3d = query_pts3d[idxs]
            query_pts2d = query_pts2d[idxs]
        if reprojection_error_diag_ratio is not None:
            reprojection_error_img = reprojection_error_diag_ratio * math.sqrt(W**2 + H**2)
        else:
            reprojection_error_img = reprojection_error
        success, pr_querycam_to_world = run_pnp(query_pts2d, query_pts3d,
                    K[i], None,
                    pnp_mode, reprojection_error_img, img_size=[W, H])
        assert success
        cam2toworlds.append(pr_querycam_to_world)
    
    cam2toworld = torch.tensor(np.stack(cam2toworlds, axis=0)).to(output_dict["xyz_1"].device)
    return cam2toworld # Bx4x4

        
def depthmap_to_leftcam_Xcam(
    batch,
    dataset):
    camera_intrinsics = batch[0]["camera_intrinsics"] # Bx3x3 
    fu = camera_intrinsics[:, 0:1, 0:1]
    fv = camera_intrinsics[:, 1:2, 1:2]
    cu = camera_intrinsics[:, 0:1, 2:3] # Bx1x1
    cv = camera_intrinsics[:, 1:2, 2:3] # Bx1x1
    if "disp" in batch[0]:
        depthmap, _ = dataset.gt_disp2depth(batch)
    else:
        depthmap = batch[0]["depth"]        
    B, H, W = depthmap.shape
    v, u = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    z_cam = depthmap.float() # BxHxW
    v = v[None].repeat(B, 1, 1).to(z_cam.device) 
    u = u[None].repeat(B, 1, 1).to(z_cam.device)
   
    y_cam = (v - cv) * z_cam / fv
    x_cam = (u - cu) * z_cam / fu
    X_cam = torch.stack((x_cam, y_cam, z_cam), dim=-1).float() # BxHxWx3

    valid_mask = (depthmap > 0.0)

    return X_cam, valid_mask


# not necessarily have second frame depth gt or camera pose
# not resized to raw yet
# return the scale of gt
def depthmap_to_leftcam(
    batch,
    dataset
):
    X_cam, valid_mask = depthmap_to_leftcam_Xcam(
        batch,
        dataset
    )
    
    return pts_scale(X_cam, valid_mask), valid_mask
   

def pts_scale(
    X_cam, # BxHxWx3
    valid_mask, # BxHxW
):
    B = X_cam.shape[0]
    X_cam = X_cam.clone()
    X_cam[~valid_mask] = float('nan')
    _norm = X_cam.norm(dim=-1).view(B, -1) # Bx(H*W)
    scale = torch.nanmedian(_norm, dim=1).values.view(-1) # (B,)
    return scale


class BaseModel:
    def __init__(self, **kwargs):
        pass
    
    def forward(self,
        input_dict: Dict,
        batch: List,
        dataset):
        raise NotImplementedError
    
    def to(self, device):
        self.predictor = self.predictor.to(device)
    
    def median_scale(self, pred, target, return_scale=False):
        mask = target == 0.
        B, H, W = mask.shape
        pred_ = pred + 0.
        pred[mask] = float('nan')
        target[mask] = float('nan')
        pred = pred.view(B, -1)
        target = target.view(B, -1)
        pred_median = torch.nanmedian(pred, dim=1).values.view(B, 1, 1)
        target_median = torch.nanmedian(target, dim=1).values.view(B, 1, 1)
        if not return_scale:
            return pred_ / pred_median * target_median
        else:
            return pred_ / pred_median * target_median, target_median/pred_median
