# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/


from typing import Optional, Dict, List
import os

import torch

from zmsf.utils.geometry import world_to_camera_coordinates, camera_to_world_coordinates, scene_flow_to_optical_flow
import zmsf.utils.path_to_mast3r
import mast3r.utils.path_to_dust3r
from dust3r.post_process import estimate_focal_knowing_depth
from .base import BaseModel, depthmap_to_leftcam, pts_scale, get_pose


class St3RModel(BaseModel):
    def __init__(self, 
        predictor, 
        weight_path: str,
        pretrained: str,
        exp_name: Optional[str],
        camera_space: Optional[bool]=False,
        first_camera: Optional[bool]=False,
        decompose: Optional[bool]=False,
        estimate_focal: Optional[bool]=False,
        **kwargs):
        super().__init__(**kwargs)
        self.camera_space = camera_space
        self.first_camera = first_camera 
        self.decompose = decompose
        if self.decompose:
            assert self.camera_space
            assert not self.first_camera

        self.estimate_focal = estimate_focal

        self.predictor = predictor
        self.pretrained = pretrained
        self.loaded_checkpoint = False

        # try to load trained checkpoint first
        saved_last = None
        if exp_name is not None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            exp_ckpt_dir = os.path.join(script_dir, '..', '..', f"lightning_logs/{exp_name}/checkpoints")
            if os.path.exists(os.path.join(exp_ckpt_dir, "last.ckpt")):
                saved_last = os.path.join(exp_ckpt_dir, "last.ckpt")
            elif os.path.exists(exp_ckpt_dir):
                saved_ckpts = sorted([file for file in os.listdir(exp_ckpt_dir) if file.endswith(".ckpt")])
                if saved_ckpts:
                    saved_last = os.path.join(exp_ckpt_dir, saved_ckpts[-1])
        if weight_path is None and saved_last is not None:
            weight_path = saved_last

        if weight_path is not None:
            state_dict = torch.load(weight_path)["state_dict"]
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("predictor."):
                    new_state_dict[key[10:]] = value
            self.predictor.load_state_dict(new_state_dict, strict=True)
            del new_state_dict 
            self.loaded_checkpoint = True
            
        # otherwise, load dust3r/mast3r weights
        if not self.loaded_checkpoint:
            self.load_pretrained_weight_predictor()

        self.exp_name = exp_name
        if self.estimate_focal:
            self.exp_name += "_focal"
    
    def load_pretrained_weight_predictor(self):    
        # the pretraining loading has to be here, cannot be in setup
        device = torch.device(f'cuda:{self.global_rank}')
        
        if self.pretrained and not self.loaded_checkpoint:
            print('Loading pretrained for: ', self.pretrained, torch.cuda.current_device(), device) 
        
            ckpt = torch.load(self.pretrained, map_location=device)
            
            # Extract model state dict
            model_state = ckpt.get('model', ckpt)  # Fallback to entire dict if 'model' key doesn't exist
            
            # Process the state dict
            new_state_dict = {}
            for key, value in model_state.items():
                new_state_dict[key] = value
                if key.startswith('dec_blocks') and not any(k.startswith('dec_blocks2') for k in model_state):
                    new_state_dict[key.replace('dec_blocks', 'dec_blocks2')] = value.clone()
                if self.predictor.infer_flow:
                    if key.startswith('downstream_head1.dpt') and not any(k.startswith('downstream_head1.dpt_flow') for k in model_state):
                        new_state_dict[key.replace('downstream_head1.dpt', 'downstream_head1.dpt_flow')] = value.clone()
                    if key.startswith('downstream_head2.dpt') and not any(k.startswith('downstream_head2.dpt_flow') for k in model_state):
                        new_state_dict[key.replace('downstream_head2.dpt', 'downstream_head2.dpt_flow')] = value.clone()


            
            # Clear original checkpoint from memory
            del ckpt
            del model_state
            
            # Update model parameters
            self.predictor.load_state_dict(new_state_dict, strict=False)
            
            # Clear processed state dict
            for key in new_state_dict:
                new_state_dict[key] = None
            del new_state_dict
            
            # Force garbage collection
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            
            self.loaded_checkpoint = True

    def get_optical_flow(self,
        batch: List[Dict],
        output_dict: Dict):
        
        if self.decompose:
            optical_flow_fwd = output_dict["flow_1"][..., :2].permute(0, 3, 1, 2)
            _, _, old_H, old_W = optical_flow_fwd.shape
            _, _, H, W = batch[0]["img_raw"].shape
            optical_flow_fwd = torch.nn.functional.interpolate(optical_flow_fwd, 
                size=(H, W), mode='bilinear',
                align_corners=True)
            optical_flow_fwd[..., :1] *= float(W) / old_W
            optical_flow_fwd[..., 1:] *= float(H) / old_H
            return optical_flow_fwd
        
        # get camera-space scene flow
        scene_flow = self.get_sf(
            batch,
            output_dict
        )
        
        ### given camera-space scene flow, get optical flow
        if "camera_intrinsics" not in batch[0]:
            f = self.get_focal(output_dict)
            B, _, H, W = batch[0]["img_raw"].shape
            intrinsics = torch.eye(3).to(batch[0]["img_raw"].device)[None].expand(B, -1, -1)
            intrinsics[:, :2,:2] *= f
            intrinsics[:, :1, 2:] = W/2.
            intrinsics[:, 1:2, 2:] = H/2.
        elif self.estimate_focal:
            f = self.get_focal(output_dict)
            intrinsics = batch[0]["camera_intrinsics"].float()
            scale_x = f/intrinsics[:, 0, 0]
            scale_y = f/intrinsics[:, 1, 1]
            intrinsics[:, 0:1] *= scale_x[:, None, None]
            intrinsics[:, 1:2] *= scale_y[:, None, None]
        else:
            # f here is strictly intrinsics matrix instead!
            intrinsics = batch[0]["camera_intrinsics"].float()
        optical_flow_fwd = scene_flow_to_optical_flow(
            scene_flow.permute(0, 2, 3, 1),
            intrinsics,
            output_dict["xyz_1"]+ 0.
        ).permute(0, 3, 1, 2) # Bx2xHxW
        _, _, old_H, old_W = optical_flow_fwd.shape
        _, _, H, W = batch[0]["img_raw"].shape
        optical_flow_fwd = torch.nn.functional.interpolate(optical_flow_fwd, 
            size=(H, W), mode='bilinear',
            align_corners=True)
        optical_flow_fwd[..., :1] *= float(W) / old_W
        optical_flow_fwd[..., 1:] *= float(H) / old_H
        
        return optical_flow_fwd
    
    def get_sf(self,
        batch: List[Dict],
        output_dict: Dict):
        if self.estimate_focal or "camera_intrinsics" not in batch[0]:
            f = self.get_focal(output_dict)
            f_fwd = f
        else: 
            f = batch[0]["camera_intrinsics"][:, 0, 0]
            f_fwd = batch[1]["camera_intrinsics"][:, 0, 0]
        #### get camera-space scene flow ###
        if self.camera_space:
            if self.first_camera:
                # get right frame point cloud in left frame space
                pcd_2 = output_dict["xyz_1"] + output_dict["flow_1"] 
                # transform right point cloud back to right frame space
                if torch.all(batch[1]["camera_pose"] == 0):
                    # get camera pose
                    K = batch[1]["camera_intrinsics"] + 0.
                    K[:, :1,:1] = f_fwd
                    K[:, 1:2, 1:2] = f_fwd        
                    camera_pose_2 = get_pose(
                        output_dict,
                        K.cpu().numpy(),
                        pnp_max_points=100000,
                        reprojection_error=5.,
                        reprojection_error_diag_ratio=None,
                        pnp_mode="cv2",
                    )
                else:
                    camera_pose_2 = batch[1]["camera_pose"] + 0.
                pcd_2 = world_to_camera_coordinates(
                            camera_to_world_coordinates(
                                pcd_2, batch[0]["camera_pose"].to(pcd_2.dtype)),
                            camera_pose_2.to(pcd_2.dtype))
                scene_flow = pcd_2 - output_dict["xyz_1"]
            else:
                scene_flow = output_dict["flow_1"] + 0.
        else:
            # get left frame point cloud in world space
            pcd_1 = camera_to_world_coordinates(
                        output_dict["xyz_1"], batch[0]["camera_pose"].to(output_dict["xyz_1"].dtype))
            # get flowed point cloud in world space
            pcd_2 = pcd_1 + output_dict["flow_1"] 
            # transform right point cloud back to right frame space
            if torch.all(batch[1]["camera_pose"] == 0):
                # get camera pose
                K = batch[1]["camera_intrinsics"] + 0.
                K[:, :1,:1] = f_fwd
                K[:, 1:2, 1:2] = f_fwd        
                camera_pose_2 = get_pose(
                    output_dict,
                    K.cpu().numpy(),
                    pnp_max_points=100000,
                    reprojection_error=5.,
                    reprojection_error_diag_ratio=None,
                    pnp_mode="cv2",
                )
            else:
                camera_pose_2 = batch[1]["camera_pose"] + 0.
            pcd_2 = world_to_camera_coordinates(
                pcd_2, camera_pose_2.to(pcd_2.dtype))
            scene_flow = pcd_2 - output_dict["xyz_1"]
        return scene_flow.permute(0, 3, 1, 2) # Bx3xHxW

    def get_focal(self,
        output_dict: Dict
    ):
        _, height, width, _ = output_dict["xyz_1"].shape
        device = output_dict["xyz_1"].device
        pp = torch.tensor((width/2, height/2), device=device)
        f = estimate_focal_knowing_depth(
                output_dict["xyz_1"], 
                pp, 
                focal_mode='weiszfeld'
            ).float()
        while len(f.shape) > 1:
            assert f.shape[-1] == 1
            f = f.squeeze(-1)
        return f

    def get_pts_sf(self,
        dataset,
        batch: List[Dict],
        output_dict: Dict,
        optical_flow: torch.Tensor
    ):
        _, _, H, W = batch[0]["img_raw"].shape
        pts1 = torch.nn.functional.interpolate(
            output_dict['xyz_1'].permute(0, 3, 1, 2),
            (H, W), mode="nearest")
        scene_flow = self.get_sf(batch, output_dict)
        sf = torch.nn.functional.interpolate(
            scene_flow,
            (H, W), mode="nearest")
        return pts1, sf
        
    def get_D1_D2(self,
        dataset,
        batch: List[Dict],
        output_dict: Dict,
        optical_flow: torch.Tensor
    ):

        """Get Disparities-> Scale Issue already handled during forwarding """
        z = output_dict['xyz_1'][..., 2]+ 0. # BxHxW
        """Get Scene Flow -> Scale Issue already handled during forwarding """
        pr_flow1 = self.get_sf(batch, output_dict) # Bx3xHxW
        
        z_fwd = output_dict['xyz_1'][..., 2] + pr_flow1[:, 2] # BxHxW
        if self.estimate_focal:
            f = self.get_focal(output_dict)
            f_fwd = f
        else: 
            f = batch[0]["camera_intrinsics"][:, 0, 0]
            f_fwd = batch[1]["camera_intrinsics"][:, 0, 0]
        disp = dataset.depth2disp(batch[0], z, f)
        disp_fwd = dataset.depth2disp(batch[1], z_fwd, f_fwd)
        disp = dataset.resize_disp_to_raw(batch[0], disp) 
        disp_fwd = dataset.resize_disp_to_raw(batch[1], disp_fwd)

        """ get depths: scaled/raw for first-> Scale Issue already handled during forwarding """
        # get depths
        # depth: scaled depth
        # depth_2: not-scaled depth
        # these would be in raw scale
        depth = output_dict['xyz_1'][..., 2]+ 0. # BxHxW
        depth_2 = output_dict["xyz_1_raw"][..., 2]+ 0. # BxHxW
        depth = dataset.resize_depth_to_raw(batch[0], depth)
        depth_2 = dataset.resize_depth_to_raw(batch[0], depth_2)


        """Resize Scene Flow to match raw"""        
        gt_img = batch[0]["img_raw"]
        B, _, H, W = gt_img.shape
        out_sceneflow = torch.nn.functional.interpolate(
            pr_flow1, size=(H, W), 
            mode="bilinear", align_corners=True) # Bx3xHxW
            
        return disp, disp_fwd, depth, depth_2, out_sceneflow #, out_sceneflow * depth_scale

    def forward(self,
        input_dict: Dict,
        batch: List,
        dataset):
         
        output_dict = self.predictor.forward(input_dict)

        if "camera_intrinsics" in input_dict["view1"]:
            gt_scale, valid_mask = depthmap_to_leftcam(batch, dataset)
            pred_scale = pts_scale(output_dict["xyz_1"], valid_mask)
            scaler = (gt_scale / pred_scale)[:, None, None, None]
            
            # keep a copy of before scale 
            output_dict["xyz_1_raw"] = output_dict["xyz_1"] + 0.
            output_dict["xyz_2_raw"] = output_dict["xyz_2"] + 0.
            output_dict["flow_1_raw"] = output_dict["flow_1"] + 0.
            output_dict["flow_2_raw"] = output_dict["flow_2"] + 0.

            output_dict["xyz_1"] *= scaler
            output_dict["xyz_2"] *= scaler
            if self.decompose:
                output_dict["flow_1"][..., 2:] *= scaler
                output_dict["flow_2"][..., 2:] *= scaler
            else:
                output_dict["flow_1"] *= scaler
                output_dict["flow_2"] *= scaler
    
        return output_dict
