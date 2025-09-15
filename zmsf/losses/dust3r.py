# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/

# The Regr3D class in this file extends the corresponding class in https://github.com/naver/dust3r/blob/main/dust3r/losses.py
# The original file is subject to the license located at https://github.com/naver/dust3r/blob/main/LICENSE


import torch

from .base import Criterion, MultiLoss, Sum
from zmsf.utils.geometry import world_to_camera_coordinates, normalize_pointcloud


class Regr3D (Criterion, MultiLoss):
    """ Ensure that all 3D points are correct.
        Asymmetric loss: view1 is supposed to be the anchor.

        P1 = RT1 @ D1
        P2 = RT2 @ D2
        loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
        loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
              = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """
    def __init__(self, criterion, norm_mode='avg_dis', gt_scale=False):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale

    def get_all_pts3d(self, input_dict, output_dict, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        gt_pts1 = world_to_camera_coordinates(input_dict["view1"]["pts3d"], input_dict["view1"]["camera_pose"]) 
        gt_pts2 = world_to_camera_coordinates(input_dict["view2"]["pts3d"], input_dict["view1"]["camera_pose"]) 
                
        valid1 = input_dict["view1"]['valid_mask'].clone()
        valid2 = input_dict["view1"]['valid_mask'].clone()
        loss_scaler_1 = 1.
        loss_scaler_2 = 1.
        if input_dict["filter_loss"]:
            use_depth_loss_1 = input_dict["view1"]["use_depth_loss"].view(-1, 1, 1).expand_as(valid1)
            use_depth_loss_2 = input_dict["view2"]["use_depth_loss"].view(-1, 1, 1).expand_as(valid2)
            valid1 &= use_depth_loss_1
            valid2 &= use_depth_loss_2
            # scale the losses 
            B = float(len(input_dict["view1"]["use_depth_loss"]))
            if torch.any(use_depth_loss_1):
                loss_scaler_1 = B / torch.sum(input_dict["view1"]["use_depth_loss"])
                
            if torch.any(use_depth_loss_2):
                loss_scaler_2 = B / torch.sum(input_dict["view2"]["use_depth_loss"])
                

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis1 = gt_pts1.norm(dim=-1)  # (B, H, W)
            dis2 = gt_pts2.norm(dim=-1)  # (B, H, W)
            valid1 = valid1 & (dis1 <= dist_clip)
            valid2 = valid2 & (dis2 <= dist_clip)

        pr_pts1 = output_dict["xyz_1"] * 1.
        pr_pts2 = output_dict["xyz_2"] * 1.

        # normalize 3d points
        if self.norm_mode:
            pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2)
        else:
            assert False, "Not allowed for now!"
        if self.norm_mode and not self.gt_scale:
            gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2)
        else:
            assert False, "Not allowed for now!"

        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, loss_scaler_1, loss_scaler_2, {}

    def compute_loss(self, input_dict, output_dict, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, loss_scaler_1, loss_scaler_2, monitoring = \
            self.get_all_pts3d(input_dict, output_dict, **kw)
        # loss on img1 side
        l1 = self.criterion(pred_pts1[mask1], gt_pts1[mask1]) * loss_scaler_1
        # loss on gt2 side
        l2 = self.criterion(pred_pts2[mask2], gt_pts2[mask2]) * loss_scaler_2
        self_name = type(self).__name__
        details = {self_name + '_pts3d_1': float(l1.mean()), self_name + '_pts3d_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)
