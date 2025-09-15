# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/


import torch
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .base import Criterion, MultiLoss, Sum, L21, L11
from .dust3r import Regr3D as Regr3D_dust3r
from zmsf.utils.geometry import world_to_camera_coordinates, camera_to_world_coordinates, normalize_pointcloud, scene_flow_to_optical_flow, world_flow_to_optical_flow, first_flow_to_optical_flow, project_point_cloud_to_image_batch, crop_image_boundaries, compose_scene_flow


class Photometric(Regr3D_dust3r):
    def __init__(self, criterion, use_flow=True, no_mask=True,
        custom_profiler=False, gaussian_renderer=False, crop_boundary=0.05):
        super().__init__(criterion)
        
        self.gaussian_renderer = gaussian_renderer
        self.crop_boundary = crop_boundary

        self.use_flow = use_flow
        self.no_mask = no_mask
        self.limit_B = 1
        
        self.custom_profiler = custom_profiler
        if self.custom_profiler:
            self.timing_events = {
                "start": torch.cuda.Event(enable_timing=True),
                "load_gt": torch.cuda.Event(enable_timing=True),
                "get_xyz": torch.cuda.Event(enable_timing=True),
                "prepare_rasterizer": torch.cuda.Event(enable_timing=True),
                "precompute_gsparam": torch.cuda.Event(enable_timing=True),
                "render_left": torch.cuda.Event(enable_timing=True),
                "go_gaussian_xyz": torch.cuda.Event(enable_timing=True),
                "render_proj_right": torch.cuda.Event(enable_timing=True),
                "render_proj_left": torch.cuda.Event(enable_timing=True),
                "go_gaussian_flow": torch.cuda.Event(enable_timing=True),
                "end": torch.cuda.Event(enable_timing=True),
            }
            assert self.use_flow
            assert self.limit_B == 1
    
    def get_all_img(self, input_dict, output_dict):
        # shape: Bx3xHxW
        # range: [-1., 1.] to [0., 1.]
        gt_left = input_dict["view1"]["img"]/2. + .5
        gt_right = input_dict["view2"]["img"]/2. + .5

        B, _, height, width = gt_left.shape

        valid1 = input_dict["view1"]["valid_mask"] # BxHxW
        valid2 = input_dict["view2"]["valid_mask"]
        loss_scaler_1 = 1.
        loss_scaler_2 = 1.
        if self.no_mask:
            valid1 = valid1.new_ones(B, height, width)
            valid2 = valid2.new_ones(B, height, width)
        if input_dict["filter_loss"]:
            use_photo_loss_1 = input_dict["view1"]["use_photo_loss"].view(-1, 1, 1).expand_as(valid1)
            use_photo_loss_2 = input_dict["view2"]["use_photo_loss"].view(-1, 1, 1).expand_as(valid2)
            valid1 = valid1 & use_photo_loss_1 
            valid2 = valid2 & use_photo_loss_2 
            # scale the losses 
            B = float(len(input_dict["view1"]["use_photo_loss"]))
            if torch.any(use_photo_loss_1):
                loss_scaler_1 = B / torch.sum(input_dict["view1"]["use_photo_loss"])
                
            if torch.any(use_photo_loss_2):
                loss_scaler_2 = B / torch.sum(input_dict["view2"]["use_photo_loss"])
            
        valid_masks = torch.cat([
            valid1, valid2
        ], dim=0)

        bg_color = output_dict["xyz_1"].new_zeros(1, 3)

        if self.custom_profiler:
            self.timing_events["load_gt"].record()
        
        # raw point clouds both in left camera coordinate
        # both of shape BxHxWx3
        xyz_left = output_dict["xyz_1"]
        xyz_right = output_dict["xyz_2"]
        # right point cloud in right camera coordinates
        xyz_right_world = camera_to_world_coordinates(
            xyz_right, 
            input_dict["view1"]["camera_pose"].float())
        xyz_right_right = world_to_camera_coordinates(
            xyz_right_world,
            input_dict["view2"]["camera_pose"].float()
        )
        if self.use_flow:
            if input_dict["decompose"]:
                fwd_flow = compose_scene_flow(
                    pts3d_left=xyz_left,
                    flow_left_to_right = output_dict["flow_1"][..., :2],
                    depth_change = output_dict["flow_1"][..., 2:],
                    intrinsics = input_dict["view2"]["camera_intrinsics"].float()
                )# BxHxWx3;                
                bwd_flow = compose_scene_flow(
                    pts3d_left=xyz_right_right,
                    flow_left_to_right = output_dict["flow_2"][..., :2],
                    depth_change = output_dict["flow_2"][..., 2:],
                    intrinsics = input_dict["view1"]["camera_intrinsics"].float()
                )

                proj_xyz_right = xyz_left + fwd_flow
                proj_xyz_left = xyz_right_right + bwd_flow
            else:
                if input_dict["camera_space"]:
                    if input_dict["first_camera"]:
                        proj_xyz_right = xyz_left + output_dict["flow_1"] # BxHxWx3
                        proj_xyz_right = world_to_camera_coordinates(
                            camera_to_world_coordinates(
                                proj_xyz_right,
                                input_dict["view1"]["camera_pose"].float() #Bx4x4
                            ),
                            input_dict["view2"]["camera_pose"].float() 
                        ) # BxHxWx3
                        proj_xyz_left = xyz_right + output_dict["flow_2"] # BxHxWx3
                    else:
                        proj_xyz_right = xyz_left + output_dict["flow_1"]
                        proj_xyz_left = xyz_right_right + output_dict["flow_2"]
                else:
                    proj_xyz_right = camera_to_world_coordinates(
                        xyz_left,
                        input_dict["view1"]["camera_pose"].float() #Bx4x4
                    )
                    proj_xyz_right += output_dict["flow_1"]
                    proj_xyz_right = world_to_camera_coordinates(
                        proj_xyz_right,
                        input_dict["view2"]["camera_pose"].float() 
                    )
                    proj_xyz_left = xyz_right_world + output_dict["flow_2"]
                    proj_xyz_left = world_to_camera_coordinates(
                        proj_xyz_left,
                        input_dict["view1"]["camera_pose"].float()
                    )
        if self.custom_profiler:
            self.timing_events["get_xyz"].record()

        if not self.gaussian_renderer:
            rendered_lefts = project_point_cloud_to_image_batch(
                xyz_left,
                gt_left,
                input_dict["view1"]["camera_intrinsics"].float()
            )
            rendered_rights = project_point_cloud_to_image_batch(
                xyz_right_right,
                gt_right,
                input_dict["view2"]["camera_intrinsics"].float()
            )
            if self.use_flow:
                projected_image_rights = project_point_cloud_to_image_batch(
                    proj_xyz_right,
                    gt_left,
                    input_dict["view2"]["camera_intrinsics"].float()
                )
                projected_image_lefts = project_point_cloud_to_image_batch(
                    proj_xyz_left,
                    gt_right,
                    input_dict["view1"]["camera_intrinsics"].float()
                )

            if self.crop_boundary > 0.:
                gt_left = crop_image_boundaries(gt_left, self.crop_boundary)
                gt_right = crop_image_boundaries(gt_right, self.crop_boundary)
                rendered_lefts = crop_image_boundaries(rendered_lefts, self.crop_boundary)
                rendered_rights = crop_image_boundaries(rendered_rights, self.crop_boundary)
                valid1 = crop_image_boundaries(valid1[:, None], self.crop_boundary)[:, 0]
                valid2 = crop_image_boundaries(valid2[:, None], self.crop_boundary)[:, 0]
                if self.use_flow:
                    projected_image_rights = crop_image_boundaries(projected_image_rights, self.crop_boundary)
                    projected_image_lefts = crop_image_boundaries(projected_image_lefts, self.crop_boundary)
            
            if not self.use_flow:
                projected_image_rights, projected_image_lefts = None, None
            return gt_left, gt_right, rendered_lefts, rendered_rights, projected_image_lefts, projected_image_rights, valid1, valid2, loss_scaler_1, loss_scaler_2, {} 
            
        else:
            assert False, "Not supporting filter loss"
            # prepare Gaussian Rasterizer
            world_view_transform = torch.eye(4, device=bg_color.device, dtype=bg_color.dtype)
            assert torch.all(input_dict["view1"]["camera_intrinsics"] == input_dict["view2"]["camera_intrinsics"])
            camera_center = bg_color.new_zeros(3,)
        
            rendered_lefts = []
            rendered_rights = []
            if self.use_flow:
                projected_image_lefts = []
                projected_image_rights = []

            for img_id in range(B):

                # Extract intrinsic parameters
                # for now assume the same fx and fy for pair of images
                full_proj_transform = input_dict["view1"]["full_proj_transform"][img_id].float() # 4x4
                tanfovx = float(input_dict["view1"]["tanfovx"][img_id])
                tanfovy = float(input_dict["view1"]["tanfovy"][img_id])
                # Set up rasterizer
                raster_settings = GaussianRasterizationSettings(
                    image_height=height,
                    image_width=width,
                    tanfovx=tanfovx,
                    tanfovy=tanfovy,
                    bg=bg_color,
                    scale_modifier=1.0,
                    viewmatrix=world_view_transform,
                    projmatrix=full_proj_transform,
                    sh_degree=output_dict["max_sh_degree"],
                    campos=camera_center,
                    prefiltered=False,
                    debug=True
                )
                rasterizer = GaussianRasterizer(raster_settings=raster_settings)
                if self.custom_profiler:
                    self.timing_events["prepare_rasterizer"].record()

                means2D_left = xyz_left.new_zeros(xyz_left[img_id][valid_masks[img_id]].shape[0], 3)
                means2D_right = xyz_right.new_zeros(xyz_right[img_id][valid_masks[B+img_id]].shape[0], 3)
                shs_left = torch.cat([
                            output_dict["features_dc_1"][img_id][valid_masks[img_id]],
                            output_dict["features_rest_1"][img_id][valid_masks[img_id]]
                        ], dim=1).contiguous()
                shs_right = torch.cat([
                            output_dict["features_dc_2"][img_id][valid_masks[B+img_id]],
                            output_dict["features_rest_2"][img_id][valid_masks[B+img_id]]
                        ], dim=1).contiguous()
                rotation_left = output_dict["rotation_1"][img_id][valid_masks[img_id]]
                rotation_right = output_dict["rotation_2"][img_id][valid_masks[B+img_id]]
                opacity_left = output_dict["opacity_1"][img_id][valid_masks[img_id]]
                opacity_right = output_dict["opacity_2"][img_id][valid_masks[B+img_id]]
                scaling_left = output_dict["scaling_1"][img_id][valid_masks[img_id]]
                scaling_right = output_dict["scaling_2"][img_id][valid_masks[B+img_id]]

                if self.custom_profiler:
                    self.timing_events["precompute_gsparam"].record()
         
                rendered_left, _ = rasterizer(
                    means3D = xyz_left[img_id][valid_masks[img_id]],
                    means2D = means2D_left,
                    shs = shs_left,
                    colors_precomp = None,
                    opacities = opacity_left,
                    scales = scaling_left,
                    rotations = rotation_left,
                    cov3D_precomp = None)
                
                rendered_lefts.append(rendered_left)
                
                if self.custom_profiler:
                    self.timing_events["render_left"].record()
                  
                rendered_right, _ = rasterizer(
                    means3D = xyz_right_right[img_id][valid_masks[B+img_id]],
                    means2D = means2D_right,
                    shs = shs_right,
                    colors_precomp = None,
                    opacities = opacity_right,
                    scales = scaling_right,
                    rotations = rotation_right,
                    cov3D_precomp = None) # 3xHxW

                rendered_rights.append(rendered_right)
                
                if self.custom_profiler:
                    self.timing_events["go_gaussian_xyz"].record()
                if self.use_flow:
                    projected_image_right, _ = rasterizer(
                        means3D = proj_xyz_right[img_id][valid_masks[img_id]],
                        means2D = means2D_left,
                        shs = shs_left,
                        colors_precomp = None,
                        opacities = opacity_left,
                        scales = scaling_left,
                        rotations = rotation_left,
                        cov3D_precomp = None)# 3xHxW
                    projected_image_rights.append(projected_image_right)
                    if self.custom_profiler:
                        self.timing_events["render_proj_right"].record()
                    
                    projected_image_left, _ = rasterizer(
                        means3D = proj_xyz_left[img_id][valid_masks[B+img_id]],
                        means2D = means2D_right,
                        shs = shs_right,
                        colors_precomp = None,
                        opacities = opacity_right,
                        scales = scaling_right,
                        rotations = rotation_right,
                        cov3D_precomp = None) # 3xHxW
                    projected_image_lefts.append(projected_image_left)

                    if self.custom_profiler:
                        self.timing_events["render_proj_left"].record()
                    
                if img_id +1 == self.limit_B:
                    break
            rendered_lefts = torch.stack(rendered_lefts, dim=0)
            rendered_rights = torch.stack(rendered_rights, dim=0)

            if self.use_flow:
                projected_image_lefts = torch.stack(projected_image_lefts, dim=0)
                projected_image_rights = torch.stack(projected_image_rights, dim=0)
            else:
                projected_image_lefts = None
                projected_image_rights = None
            
            if self.crop_boundary > 0.:
                gt_left = crop_image_boundaries(gt_left, self.crop_boundary)
                gt_right = crop_image_boundaries(gt_right, self.crop_boundary)
                rendered_lefts = crop_image_boundaries(rendered_lefts, self.crop_boundary)
                rendered_rights = crop_image_boundaries(rendered_rights, self.crop_boundary)
                valid1 = crop_image_boundaries(valid1[:, None], self.crop_boundary)[:, 0]
                valid2 = crop_image_boundaries(valid2[:, None], self.crop_boundary)[:, 0]
                if self.use_flow:
                    projected_image_rights = crop_image_boundaries(projected_image_rights, self.crop_boundary)
                    projected_image_lefts = crop_image_boundaries(projected_image_lefts, self.crop_boundary)

            if self.custom_profiler:
                self.timing_events["go_gaussian_flow"].record()

            return gt_left[:self.limit_B], gt_right[:self.limit_B], rendered_lefts, rendered_rights, projected_image_lefts, projected_image_rights, valid1[:self.limit_B], valid2[:self.limit_B], loss_scaler_1, loss_scaler_2, {}             
        
    def compute_loss(self, input_dict, output_dict, **kw):
        # Start CUDA timing
        
        if self.custom_profiler:
            self.timing_events["start"].record()

        gt_left, gt_right, rendered_lefts, rendered_rights, projected_image_lefts, projected_image_rights, mask1, mask2, loss_scaler_1, loss_scaler_2, monitoring = \
            self.get_all_img(input_dict, output_dict, **kw)

        # below all have shape Bx3xHxW
        # first permute to BxHxWx3 then could apply BxHxW masks
        # then could pass into criterions which do operations with dim=-1
        gt_left = gt_left.permute(0, 2, 3, 1)[mask1]
        gt_right = gt_right.permute(0, 2, 3, 1)[mask2]
        rendered_lefts = rendered_lefts.permute(0, 2, 3, 1)[mask1]
        rendered_rights = rendered_rights.permute(0, 2, 3, 1)[mask2]
        if self.use_flow:
            projected_image_lefts = projected_image_lefts.permute(0, 2, 3, 1)[mask1]
            projected_image_rights = projected_image_rights.permute(0, 2, 3, 1)[mask2] 
        
        l1 = self.criterion(gt_left, rendered_lefts) * loss_scaler_1
        l2 = self.criterion(gt_right, rendered_rights) * loss_scaler_2
        if self.use_flow:
            l1_f = self.criterion(gt_left, projected_image_lefts) * loss_scaler_1
            l2_f = self.criterion(gt_right, projected_image_rights) * loss_scaler_2 
            if input_dict["unidir"]:
                l1_f *= 0.
                l2_f *= 0.
        
        self_name = type(self).__name__
        details = {
            self_name + '_1': float(l1.mean()), 
            self_name + '_2': float(l2.mean()),
        }
        if self.use_flow:
            details[self_name + '_flow_1'] = float(l1_f.mean())
            details[self_name + '_flow_2'] = float(l2_f.mean())
        
        if self.custom_profiler:
            self.timing_events["end"].record()

            # Synchronize CUDA events
            torch.cuda.synchronize()

            # Calculate elapsed time
            details = {}
            details["load_gt"] = self.timing_events["start"].elapsed_time(
                self.timing_events["load_gt"])
            details["get_xyz"] = self.timing_events["load_gt"].elapsed_time(
                self.timing_events["get_xyz"])
            details["prepare_rasterizer"] = self.timing_events["get_xyz"].elapsed_time(
                self.timing_events["prepare_rasterizer"])
            details["precompute_gsparam"] = self.timing_events["prepare_rasterizer"].elapsed_time(
                self.timing_events["precompute_gsparam"])
            details["render_left"] = self.timing_events["precompute_gsparam"].elapsed_time(
                self.timing_events["render_left"])
            details["render_right"] = self.timing_events["render_left"].elapsed_time(
                self.timing_events["go_gaussian_xyz"])
            details["render_proj_right"] = self.timing_events["go_gaussian_xyz"].elapsed_time(
                self.timing_events["render_proj_right"])
            details["render_proj_left"] = self.timing_events["render_proj_right"].elapsed_time(
                self.timing_events["render_proj_left"])
            details["go_gaussian_flow"] = self.timing_events["render_proj_left"].elapsed_time(
                self.timing_events["go_gaussian_flow"])
            details["criterion"] = self.timing_events["go_gaussian_flow"].elapsed_time(
                self.timing_events["end"])
            
            # Add elapsed time to details
            print(details, flush=True)

        if self.use_flow:
            return Sum(
                (l1, mask1), 
                (l2, mask2),
                (l1_f, mask1), 
                (l2_f, mask2),
                ), (details | monitoring)

        else:
            return Sum(
                (l1, mask1), 
                (l2, mask2),
                ), (details | monitoring)


class Regr3D_flow(Regr3D_dust3r):
    def __init__(self, criterion):
        super().__init__(criterion)

    def get_all_pts3d(self, input_dict, output_dict):
        gt_flow1 = input_dict["view1"]["flow_3d"].clone()
        gt_flow2 = input_dict["view2"]["flow_3d"].clone()
        
        valid1 = input_dict["view1"]["mask_3d"].clone()
        valid2 = input_dict["view2"]["mask_3d"].clone()
        loss_scaler_1 = 1.
        loss_scaler_2 = 1.
        if input_dict["filter_loss"]:
            use_3d_flow_loss_1 = input_dict["view1"]["use_3d_flow_loss"].view(-1, 1, 1).expand_as(valid1)
            use_3d_flow_loss_2 = input_dict["view2"]["use_3d_flow_loss"].view(-1, 1, 1).expand_as(valid2)
            valid1 &= use_3d_flow_loss_1
            valid2 &= use_3d_flow_loss_2
            # scale the losses 
            B = float(len(input_dict["view1"]["use_3d_flow_loss"]))
            if torch.any(use_3d_flow_loss_1):
                loss_scaler_1 = B / torch.sum(input_dict["view1"]["use_3d_flow_loss"])
                
            if torch.any(use_3d_flow_loss_2):
                loss_scaler_2 = B / torch.sum(input_dict["view2"]["use_3d_flow_loss"])
        if input_dict["filter_metric"]:
            use_3d_flow_loss_1 = input_dict["view1"]["is_metric_scale"].view(-1, 1, 1).expand_as(valid1)
            use_3d_flow_loss_2 = input_dict["view2"]["is_metric_scale"].view(-1, 1, 1).expand_as(valid2)
            valid1 &= use_3d_flow_loss_1
            valid2 &= use_3d_flow_loss_2
            # scale the losses 
            B = float(len(input_dict["view1"]["is_metric_scale"]))
            if torch.any(use_3d_flow_loss_1):
                loss_scaler_1 = B / torch.sum(input_dict["view1"]["is_metric_scale"])
                
            if torch.any(use_3d_flow_loss_2):
                loss_scaler_2 = B / torch.sum(input_dict["view2"]["is_metric_scale"])
        
        if input_dict["decompose"]:
            xyz_left = output_dict["xyz_1"]
            xyz_right = output_dict["xyz_2"]
            # right point cloud in right camera coordinates
            xyz_right_world = camera_to_world_coordinates(
                xyz_right, 
                input_dict["view1"]["camera_pose"].float())
            xyz_right_right = world_to_camera_coordinates(
                xyz_right_world,
                input_dict["view2"]["camera_pose"].float()
            )
            
            pr_flow1  = compose_scene_flow(
                pts3d_left=xyz_left,
                flow_left_to_right = output_dict["flow_1"][..., :2],
                depth_change = output_dict["flow_1"][..., 2:],
                intrinsics = input_dict["view2"]["camera_intrinsics"].float()
            )# BxHxWx3;                
            pr_flow2 = compose_scene_flow(
                pts3d_left=xyz_right_right,
                flow_left_to_right = output_dict["flow_2"][..., :2],
                depth_change = output_dict["flow_2"][..., 2:],
                intrinsics = input_dict["view1"]["camera_intrinsics"].float()
            )
        else:
            pr_flow1 = output_dict["flow_1"] * 1.
            pr_flow2 = output_dict["flow_2"] * 1.

        return gt_flow1, gt_flow2, pr_flow1, pr_flow2, valid1, valid2, loss_scaler_1, loss_scaler_2, {}

    def compute_loss(self, input_dict, output_dict, **kw):
        gt_flow1, gt_flow2, pred_flow1, pred_flow2, mask1, mask2, loss_scaler_1, loss_scaler_2, monitoring = \
            self.get_all_pts3d(input_dict, output_dict, **kw)
        
        # loss on img1 side
        pred_flow1 = pred_flow1[mask1]
        gt_flow1 = gt_flow1[mask1]
        l1 = self.criterion(pred_flow1, gt_flow1) * loss_scaler_1

        # loss on img2 side
        pred_flow2 = pred_flow2[mask2]
        gt_flow2 = gt_flow2[mask2]
        l2 = self.criterion(pred_flow2, gt_flow2) * loss_scaler_2
        if input_dict["unidir"]:
            l2 *= 0.

        self_name = type(self).__name__
        details = {self_name + '_flow3d_1': float(l1.mean()), self_name + '_flow3d_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)


class Regr3D_flow_norm(Regr3D_dust3r):
    def __init__(self, criterion):
        super().__init__(criterion)
        self.norm_mode = 'avg_dis'
        self.gt_scale = False

    def get_all_pts3d(self, input_dict, output_dict):
        gt_flow1 = input_dict["view1"]["flow_3d"].clone()
        gt_flow2 = input_dict["view2"]["flow_3d"].clone()
        
        valid1 = input_dict["view1"]["mask_3d"].clone()
        valid2 = input_dict["view2"]["mask_3d"].clone()
        loss_scaler_1 = 1.
        loss_scaler_2 = 1.
        if input_dict["filter_loss"]:
            use_3d_flow_loss_1 = input_dict["view1"]["use_3d_flow_loss"].view(-1, 1, 1).expand_as(valid1)
            use_3d_flow_loss_2 = input_dict["view2"]["use_3d_flow_loss"].view(-1, 1, 1).expand_as(valid2)
            valid1 &= use_3d_flow_loss_1
            valid2 &= use_3d_flow_loss_2
            # scale the losses 
            B = float(len(input_dict["view1"]["use_3d_flow_loss"]))
            if torch.any(use_3d_flow_loss_1):
                loss_scaler_1 = B / torch.sum(input_dict["view1"]["use_3d_flow_loss"])
                
            if torch.any(use_3d_flow_loss_2):
                loss_scaler_2 = B / torch.sum(input_dict["view2"]["use_3d_flow_loss"]) 

        if input_dict["decompose"]:
            xyz_left = output_dict["xyz_1"]
            xyz_right = output_dict["xyz_2"]
            # right point cloud in right camera coordinates
            xyz_right_world = camera_to_world_coordinates(
                xyz_right, 
                input_dict["view1"]["camera_pose"].float())
            xyz_right_right = world_to_camera_coordinates(
                xyz_right_world,
                input_dict["view2"]["camera_pose"].float()
            )
            
            pr_flow1  = compose_scene_flow(
                pts3d_left=xyz_left,
                flow_left_to_right = output_dict["flow_1"][..., :2],
                depth_change = output_dict["flow_1"][..., 2:],
                intrinsics = input_dict["view2"]["camera_intrinsics"].float()
            )# BxHxWx3;                
            pr_flow2 = compose_scene_flow(
                pts3d_left=xyz_right_right,
                flow_left_to_right = output_dict["flow_2"][..., :2],
                depth_change = output_dict["flow_2"][..., 2:],
                intrinsics = input_dict["view1"]["camera_intrinsics"].float()
            )
        else:
            pr_flow1 = output_dict["flow_1"] * 1.
            pr_flow2 = output_dict["flow_2"] * 1.

        pr_flow1, pr_flow2 = normalize_pointcloud(pr_flow1, pr_flow2, self.norm_mode, valid1, valid2)
        gt_flow1, gt_flow2 = normalize_pointcloud(gt_flow1, gt_flow2, self.norm_mode, valid1, valid2)
        
        return gt_flow1, gt_flow2, pr_flow1, pr_flow2, valid1, valid2, loss_scaler_1, loss_scaler_2, {}

    def compute_loss(self, input_dict, output_dict, **kw):
        gt_flow1, gt_flow2, pred_flow1, pred_flow2, mask1, mask2, loss_scaler_1, loss_scaler_2, monitoring = \
            self.get_all_pts3d(input_dict, output_dict, **kw)
        
        # loss on img1 side
        pred_flow1 = pred_flow1[mask1]
        gt_flow1 = gt_flow1[mask1]
        l1 = self.criterion(pred_flow1, gt_flow1) * loss_scaler_1

        # loss on img2 side
        pred_flow2 = pred_flow2[mask2]
        gt_flow2 = gt_flow2[mask2]
        l2 = self.criterion(pred_flow2, gt_flow2) * loss_scaler_2
        if input_dict["unidir"]:
            l2 *= 0.

        self_name = type(self).__name__
        details = {self_name + '_flow3d_norm_1': float(l1.mean()), self_name + '_flow3d_norm_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)


class Regr3D_pts_flow_norm(Regr3D_dust3r):
    def __init__(self, criterion, detach_factor=False):
        super().__init__(criterion)
        self.norm_mode = 'avg_dis'
        self.gt_scale = False
        self.detach_factor = detach_factor

    def get_all_pts3d(self, input_dict, output_dict):
        gt_pts1 = world_to_camera_coordinates(input_dict["view1"]["pts3d"], input_dict["view1"]["camera_pose"]) 
        gt_pts2 = world_to_camera_coordinates(input_dict["view2"]["pts3d"], input_dict["view1"]["camera_pose"]) 
        gt_flow1 = input_dict["view1"]["flow_3d"].clone()
        gt_flow2 = input_dict["view2"]["flow_3d"].clone()
        valid1 = input_dict["view1"]["mask_3d"] & input_dict["view1"]['valid_mask']
        valid2 = input_dict["view2"]["mask_3d"] & input_dict["view2"]['valid_mask']
        loss_scaler_1 = 1.
        loss_scaler_2 = 1.
        if input_dict["filter_loss"]:
            use_depth_loss_1 = input_dict["view1"]["use_depth_loss"].view(-1, 1, 1).expand_as(valid1)
            use_depth_loss_2 = input_dict["view2"]["use_depth_loss"].view(-1, 1, 1).expand_as(valid2)
            use_3d_flow_loss_1 = input_dict["view1"]["use_3d_flow_loss"].view(-1, 1, 1).expand_as(valid1)
            use_3d_flow_loss_2 = input_dict["view2"]["use_3d_flow_loss"].view(-1, 1, 1).expand_as(valid2)
            valid1 = valid1 & use_depth_loss_1 & use_3d_flow_loss_1 
            if input_dict["unidir"]:
                valid2 = valid2 & use_depth_loss_2
            else:
                valid2 = valid2 & use_depth_loss_2 & use_3d_flow_loss_2 
            # scale the losses 
            B = float(len(input_dict["view1"]["use_3d_flow_loss"]))
            if torch.any(use_3d_flow_loss_1 & use_depth_loss_1):
                loss_scaler_1 = B / torch.sum(input_dict["view1"]["use_3d_flow_loss"] & input_dict["view1"]["use_depth_loss"])
                
            if input_dict["unidir"]:
                if torch.any(use_depth_loss_2):
                    loss_scaler_2 = B / torch.sum(input_dict["view2"]["use_depth_loss"])
            else:
                if torch.any(use_3d_flow_loss_2 & use_depth_loss_2):
                    loss_scaler_2 = B / torch.sum(input_dict["view2"]["use_3d_flow_loss"] & input_dict["view2"]["use_depth_loss"])
           
        pr_pts1 = output_dict["xyz_1"] * 1.
        pr_pts2 = output_dict["xyz_2"] * 1.
        
        if input_dict["decompose"]:
            xyz_left = output_dict["xyz_1"]
            xyz_right = output_dict["xyz_2"]
            # right point cloud in right camera coordinates
            xyz_right_world = camera_to_world_coordinates(
                xyz_right, 
                input_dict["view1"]["camera_pose"].float())
            xyz_right_right = world_to_camera_coordinates(
                xyz_right_world,
                input_dict["view2"]["camera_pose"].float()
            )
            
            pr_flow1  = compose_scene_flow(
                pts3d_left=xyz_left,
                flow_left_to_right = output_dict["flow_1"][..., :2],
                depth_change = output_dict["flow_1"][..., 2:],
                intrinsics = input_dict["view2"]["camera_intrinsics"].float()
            )# BxHxWx3;                
            pr_flow2 = compose_scene_flow(
                pts3d_left=xyz_right_right,
                flow_left_to_right = output_dict["flow_2"][..., :2],
                depth_change = output_dict["flow_2"][..., 2:],
                intrinsics = input_dict["view1"]["camera_intrinsics"].float()
            )
        else:
            pr_flow1 = output_dict["flow_1"] * 1.
            pr_flow2 = output_dict["flow_2"] * 1.

        pr_pts1, pr_pts2, pr_factor = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2, ret_factor=True)
        if self.detach_factor:
            pr_factor = pr_factor.detach()
        pr_flow1, pr_flow2 = normalize_pointcloud(pr_flow1, pr_flow2, self.norm_mode, valid1, valid2, norm_factor=pr_factor)
        gt_pts1, gt_pts2, gt_factor = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2, ret_factor=True)
        if self.detach_factor:
            gt_factor = gt_factor.detach()
        gt_flow1, gt_flow2 = normalize_pointcloud(gt_flow1, gt_flow2, self.norm_mode, valid1, valid2, norm_factor=gt_factor)
        
        return gt_pts1, gt_pts2, gt_flow1, gt_flow2, pr_pts1, pr_pts2, pr_flow1, pr_flow2, valid1, valid2, loss_scaler_1, loss_scaler_2, {}

    def compute_loss(self, input_dict, output_dict, **kw):
        gt_pts1, gt_pts2, gt_flow1, gt_flow2, pred_pts1, pred_pts2, pred_flow1, pred_flow2, mask1, mask2, loss_scaler_1, loss_scaler_2, monitoring = \
            self.get_all_pts3d(input_dict, output_dict, **kw)
        
        # loss on img1 side
        pred_pts1 = pred_pts1[mask1]
        gt_pts1 = gt_pts1[mask1]
        l1_pts = self.criterion(pred_pts1, gt_pts1) * loss_scaler_1
        pred_flow1 = pred_flow1[mask1]
        gt_flow1 = gt_flow1[mask1]
        l1_flow = self.criterion(pred_flow1, gt_flow1) * loss_scaler_1

        # loss on img2 side
        pred_pts2 = pred_pts2[mask2]
        gt_pts2 = gt_pts2[mask2]
        l2_pts = self.criterion(pred_pts2, gt_pts2) * loss_scaler_2
        pred_flow2 = pred_flow2[mask2]
        gt_flow2 = gt_flow2[mask2]
        l2_flow = self.criterion(pred_flow2, gt_flow2) * loss_scaler_2
        if input_dict["unidir"]:
            l2_flow *= 0.

        self_name = type(self).__name__
        details = {
            self_name + '_pts_norm_1': float(l1_pts.mean()), 
            self_name + '_flow3d_norm_1': float(l1_flow.mean()), 
            self_name + '_pts_norm_2': float(l2_pts.mean()),
            self_name + '_flow3d_norm_2': float(l2_flow.mean())
        }
        return Sum(
            (l1_pts, mask1),
            (l1_flow, mask1),
            (l2_pts, mask2), 
            (l2_flow, mask2)), (details | monitoring)


class Regr3D_pts_adap(Regr3D_dust3r):
    def __init__(self, criterion, detach_factor=False):
        super().__init__(criterion)
        self.norm_mode = 'avg_dis'
        self.gt_scale = False
        self.detach_factor = detach_factor

    def get_all_pts3d(self, input_dict, output_dict):
        gt_pts1 = world_to_camera_coordinates(input_dict["view1"]["pts3d"], input_dict["view1"]["camera_pose"]) 
        gt_pts2 = world_to_camera_coordinates(input_dict["view2"]["pts3d"], input_dict["view1"]["camera_pose"]) 
        valid1 = input_dict["view1"]['valid_mask']
        valid2 = input_dict["view2"]['valid_mask']
        loss_scaler_1 = 1.
        loss_scaler_2 = 1.
        if input_dict["filter_loss"]:
            use_depth_loss_1 = input_dict["view1"]["use_depth_loss"].view(-1, 1, 1).expand_as(valid1)
            use_depth_loss_2 = input_dict["view2"]["use_depth_loss"].view(-1, 1, 1).expand_as(valid2)
            valid1 = valid1 & use_depth_loss_1  
            valid2 = valid2 & use_depth_loss_2 
            # scale the losses 
            B = float(len(input_dict["view1"]["use_depth_loss"]))
            if torch.any(use_depth_loss_1):
                loss_scaler_1 = B / torch.sum(input_dict["view1"]["use_depth_loss"])
                
            if torch.any(use_depth_loss_2):
                loss_scaler_2 = B / torch.sum(input_dict["view2"]["use_depth_loss"])
           
        pr_pts1 = output_dict["xyz_1"] * 1.
        pr_pts2 = output_dict["xyz_2"] * 1.
        
        is_metric_1 = input_dict["view1"]["is_metric_scale"].view(-1, 1, 1, 1)
        is_metric_2 = input_dict["view2"]["is_metric_scale"].view(-1, 1, 1, 1)
        assert torch.all(~torch.logical_xor(is_metric_1, is_metric_2))

        _, _, pr_factor = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2, ret_factor=True)
        if self.detach_factor:
            pr_factor = pr_factor.detach()
        pr_factor[is_metric_1] = 1.
        pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2, norm_factor=pr_factor)
        
        _, _, gt_factor = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2, ret_factor=True)
        if self.detach_factor:
            gt_factor = gt_factor.detach()
        gt_factor[is_metric_1] = 1.
        gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2, norm_factor=gt_factor)
        
        return gt_pts1, gt_pts2, pr_pts1, pr_pts2, valid1, valid2, loss_scaler_1, loss_scaler_2, {}

    def compute_loss(self, input_dict, output_dict, **kw):
        gt_pts1, gt_pts2, pred_pts1, pred_pts2, mask1, mask2, loss_scaler_1, loss_scaler_2, monitoring = \
            self.get_all_pts3d(input_dict, output_dict, **kw)
        
        # loss on img1 side
        pred_pts1 = pred_pts1[mask1]
        gt_pts1 = gt_pts1[mask1]
        l1_pts = self.criterion(pred_pts1, gt_pts1) * loss_scaler_1
        
        # loss on img2 side
        pred_pts2 = pred_pts2[mask2]
        gt_pts2 = gt_pts2[mask2]
        l2_pts = self.criterion(pred_pts2, gt_pts2) * loss_scaler_2
        
        self_name = type(self).__name__
        details = {
            self_name + '_pts_norm_1': float(l1_pts.mean()), 
            self_name + '_pts_norm_2': float(l2_pts.mean()),
        }
        return Sum(
            (l1_pts, mask1),
            (l2_pts, mask2), 
            ), (details | monitoring)


class Regr3D_pts_flow_adap(Regr3D_dust3r):
    def __init__(self, criterion, detach_factor=False):
        super().__init__(criterion)
        self.norm_mode = 'avg_dis'
        self.gt_scale = False
        self.detach_factor = detach_factor

    def get_all_pts3d(self, input_dict, output_dict):
        gt_pts1 = world_to_camera_coordinates(input_dict["view1"]["pts3d"], input_dict["view1"]["camera_pose"]) 
        gt_pts2 = world_to_camera_coordinates(input_dict["view2"]["pts3d"], input_dict["view1"]["camera_pose"]) 
        gt_flow1 = input_dict["view1"]["flow_3d"].clone()
        gt_flow2 = input_dict["view2"]["flow_3d"].clone()
        valid1 = input_dict["view1"]["mask_3d"] & input_dict["view1"]['valid_mask']
        valid2 = input_dict["view2"]["mask_3d"] & input_dict["view2"]['valid_mask']
        loss_scaler_1 = 1.
        loss_scaler_2 = 1.
        if input_dict["filter_loss"]:
            use_depth_loss_1 = input_dict["view1"]["use_depth_loss"].view(-1, 1, 1).expand_as(valid1)
            use_depth_loss_2 = input_dict["view2"]["use_depth_loss"].view(-1, 1, 1).expand_as(valid2)
            use_3d_flow_loss_1 = input_dict["view1"]["use_3d_flow_loss"].view(-1, 1, 1).expand_as(valid1)
            use_3d_flow_loss_2 = input_dict["view2"]["use_3d_flow_loss"].view(-1, 1, 1).expand_as(valid2)
            valid1 = valid1 & use_depth_loss_1 & use_3d_flow_loss_1 
            if input_dict["unidir"]:
                valid2 = valid2 & use_depth_loss_2
            else:
                valid2 = valid2 & use_depth_loss_2 & use_3d_flow_loss_2 
            # scale the losses 
            B = float(len(input_dict["view1"]["use_3d_flow_loss"]))
            if torch.any(use_3d_flow_loss_1 & use_depth_loss_1):
                loss_scaler_1 = B / torch.sum(input_dict["view1"]["use_3d_flow_loss"] & input_dict["view1"]["use_depth_loss"])
                
            if input_dict["unidir"]:
                if torch.any(use_depth_loss_2):
                    loss_scaler_2 = B / torch.sum(input_dict["view2"]["use_depth_loss"])
            else:
                if torch.any(use_3d_flow_loss_2 & use_depth_loss_2):
                    loss_scaler_2 = B / torch.sum(input_dict["view2"]["use_3d_flow_loss"] & input_dict["view2"]["use_depth_loss"])
           
        pr_pts1 = output_dict["xyz_1"] * 1.
        pr_pts2 = output_dict["xyz_2"] * 1.
        
        if input_dict["decompose"]:
            xyz_left = output_dict["xyz_1"]
            xyz_right = output_dict["xyz_2"]
            # right point cloud in right camera coordinates
            xyz_right_world = camera_to_world_coordinates(
                xyz_right, 
                input_dict["view1"]["camera_pose"].float())
            xyz_right_right = world_to_camera_coordinates(
                xyz_right_world,
                input_dict["view2"]["camera_pose"].float()
            )
            
            pr_flow1  = compose_scene_flow(
                pts3d_left=xyz_left,
                flow_left_to_right = output_dict["flow_1"][..., :2],
                depth_change = output_dict["flow_1"][..., 2:],
                intrinsics = input_dict["view2"]["camera_intrinsics"].float()
            )# BxHxWx3;                
            pr_flow2 = compose_scene_flow(
                pts3d_left=xyz_right_right,
                flow_left_to_right = output_dict["flow_2"][..., :2],
                depth_change = output_dict["flow_2"][..., 2:],
                intrinsics = input_dict["view1"]["camera_intrinsics"].float()
            )
        else:
            pr_flow1 = output_dict["flow_1"] * 1.
            pr_flow2 = output_dict["flow_2"] * 1.

        is_metric_1 = input_dict["view1"]["is_metric_scale"].view(-1, 1, 1, 1)
        is_metric_2 = input_dict["view2"]["is_metric_scale"].view(-1, 1, 1, 1)
        assert torch.all(~torch.logical_xor(is_metric_1, is_metric_2))

        _, _, pr_factor = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2, ret_factor=True)
        if self.detach_factor:
            pr_factor = pr_factor.detach()
        pr_factor[is_metric_1] = 1.
        pr_pts1, pr_pts2 = normalize_pointcloud(pr_pts1, pr_pts2, self.norm_mode, valid1, valid2, norm_factor=pr_factor)
        pr_flow1, pr_flow2 = normalize_pointcloud(pr_flow1, pr_flow2, self.norm_mode, valid1, valid2, norm_factor=pr_factor)
        
        _, _, gt_factor = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2, ret_factor=True)
        if self.detach_factor:
            gt_factor = gt_factor.detach()
        gt_factor[is_metric_1] = 1.
        gt_pts1, gt_pts2 = normalize_pointcloud(gt_pts1, gt_pts2, self.norm_mode, valid1, valid2, norm_factor=gt_factor)
        gt_flow1, gt_flow2 = normalize_pointcloud(gt_flow1, gt_flow2, self.norm_mode, valid1, valid2, norm_factor=gt_factor)   

        return gt_pts1, gt_pts2, gt_flow1, gt_flow2, pr_pts1, pr_pts2, pr_flow1, pr_flow2, valid1, valid2, loss_scaler_1, loss_scaler_2, {}

    def compute_loss(self, input_dict, output_dict, **kw):
        gt_pts1, gt_pts2, gt_flow1, gt_flow2, pred_pts1, pred_pts2, pred_flow1, pred_flow2, mask1, mask2, loss_scaler_1, loss_scaler_2, monitoring = \
            self.get_all_pts3d(input_dict, output_dict, **kw)
        
        # loss on img1 side
        pred_pts1 = pred_pts1[mask1]
        gt_pts1 = gt_pts1[mask1]
        l1_pts = self.criterion(pred_pts1, gt_pts1) * loss_scaler_1
        pred_flow1 = pred_flow1[mask1]
        gt_flow1 = gt_flow1[mask1]
        l1_flow = self.criterion(pred_flow1, gt_flow1) * loss_scaler_1

        # loss on img2 side
        pred_pts2 = pred_pts2[mask2]
        gt_pts2 = gt_pts2[mask2]
        l2_pts = self.criterion(pred_pts2, gt_pts2) * loss_scaler_2
        pred_flow2 = pred_flow2[mask2]
        gt_flow2 = gt_flow2[mask2]
        l2_flow = self.criterion(pred_flow2, gt_flow2) * loss_scaler_2
        if input_dict["unidir"]:
            l2_flow *= 0.

        self_name = type(self).__name__
        details = {
            self_name + '_pts_norm_1': float(l1_pts.mean()), 
            self_name + '_flow3d_norm_1': float(l1_flow.mean()), 
            self_name + '_pts_norm_2': float(l2_pts.mean()),
            self_name + '_flow3d_norm_2': float(l2_flow.mean())
        }
        return Sum(
            (l1_pts, mask1),
            (l1_flow, mask1),
            (l2_pts, mask2), 
            (l2_flow, mask2)), (details | monitoring)


class Regr3D_flow_2D(Regr3D_dust3r):
    def __init__(self, criterion):
        super().__init__(criterion)

    def get_all_pts3d(self, input_dict, output_dict):
        gt_flow1 = input_dict["view1"]["flow_2d"].clone()
        gt_flow2 = input_dict["view2"]["flow_2d"].clone()
        valid1 = (input_dict["view1"]["mask_3d"] *0. + 1.).bool()
        valid2 = (input_dict["view2"]["mask_3d"] *0. + 1.).bool()
        assert torch.all(valid1) and torch.all(valid2)
        loss_scaler_1 = 1.
        loss_scaler_2 = 1.
        if input_dict["filter_loss"]:
            use_2d_flow_loss_1 = input_dict["view1"]["use_2d_flow_loss"].view(-1, 1, 1).expand_as(valid1)
            use_2d_flow_loss_2 = input_dict["view2"]["use_2d_flow_loss"].view(-1, 1, 1).expand_as(valid2)
            valid1 &= use_2d_flow_loss_1 
            valid2 &= use_2d_flow_loss_2 
            # scale the losses 
            B = float(len(input_dict["view1"]["use_2d_flow_loss"]))
            if torch.any(use_2d_flow_loss_1):
                loss_scaler_1 = B / torch.sum(input_dict["view1"]["use_2d_flow_loss"])
                
            if torch.any(use_2d_flow_loss_2):
                loss_scaler_2 = B / torch.sum(input_dict["view2"]["use_2d_flow_loss"])

        #################### compute optical flow by projection ####################
        if input_dict["decompose"]:
            pr_flow1 = output_dict["flow_1"][..., :2]
            pr_flow2 = output_dict["flow_2"][..., :2]
        else:    
            if input_dict["camera_space"]:
                if input_dict["first_camera"]:
                    pr_flow1, pr_flow2 = first_flow_to_optical_flow(
                        output_dict["flow_1"], output_dict["flow_2"], #BxHxWx3
                        output_dict["xyz_1"], output_dict["xyz_2"],#BxHxWx3
                        input_dict["view1"]["camera_intrinsics"].float(), input_dict["view2"]["camera_intrinsics"].float(),#Bx3x3
                        input_dict["view1"]["camera_pose"].float(), #Bx4x4
                        input_dict["view2"]["camera_pose"].float() #Bx4x4
                    )
                else:
                    pr_flow1 = scene_flow_to_optical_flow(
                                output_dict["flow_1"], #BxHxWx3
                                input_dict["view1"]["camera_intrinsics"].float(), #Bx3x3
                                output_dict["xyz_1"], #BxHxWx3
                            ) #BxHxWx2
                    pr_flow2 = scene_flow_to_optical_flow(
                                output_dict["flow_2"],
                                input_dict["view2"]["camera_intrinsics"].float(), #Bx3x3
                                world_to_camera_coordinates(
                                    camera_to_world_coordinates(
                                        output_dict["xyz_2"], #BxHxWx3
                                        input_dict["view1"]["camera_pose"].float() #Bx4x4
                                    ), input_dict["view2"]["camera_pose"].float() #Bx4x4
                                )                        
                            ) #BxHxWx2
            else:
                pr_flow1 = world_flow_to_optical_flow(
                    output_dict["flow_1"], #BxHxWx3
                    output_dict["xyz_1"], #BxHxWx3
                    input_dict["view1"]["camera_intrinsics"].float(), #Bx3x3
                    input_dict["view1"]["camera_pose"].float(), #Bx4x4
                    input_dict["view2"]["camera_pose"].float() #Bx4x4
                )
                pr_flow2 = world_flow_to_optical_flow(
                    output_dict["flow_2"], #BxHxWx3
                    world_to_camera_coordinates(
                                camera_to_world_coordinates(
                                    output_dict["xyz_2"], #BxHxWx3
                                    input_dict["view1"]["camera_pose"].float() #Bx4x4
                                ), input_dict["view2"]["camera_pose"].float() #Bx4x4
                            ),
                    input_dict["view2"]["camera_intrinsics"].float(), #Bx3x3
                    input_dict["view2"]["camera_pose"].float(), #Bx4x4
                    input_dict["view1"]["camera_pose"].float() #Bx4x4
                ) 
                
        return gt_flow1, gt_flow2, pr_flow1, pr_flow2, valid1, valid2, loss_scaler_1, loss_scaler_2, {}

    def compute_loss(self, input_dict, output_dict, **kw):
        gt_flow1, gt_flow2, pred_flow1, pred_flow2, mask1, mask2, loss_scaler_1, loss_scaler_2, monitoring = \
            self.get_all_pts3d(input_dict, output_dict, **kw)
        
        # loss on img1 side
        pred_flow1 = pred_flow1[mask1]
        gt_flow1 = gt_flow1[mask1]
        l1 = self.criterion(pred_flow1, gt_flow1) * loss_scaler_1

        # loss on img2 side
        pred_flow2 = pred_flow2[mask2]
        gt_flow2 = gt_flow2[mask2]
        l2 = self.criterion(pred_flow2, gt_flow2) * loss_scaler_2
        if input_dict["unidir"]:
            l2 *= 0.

        self_name = type(self).__name__
        details = {self_name + '_flow2d_1': float(l1.mean()), self_name + '_flow2d_2': float(l2.mean())}
        return Sum((l1, mask1), (l2, mask2)), (details | monitoring)


class ConfLoss_pts_flow(MultiLoss):
    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')
    
    def get_name(self):
        return f'ConfLoss_pts_flow({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, input_dict, output_dict, **kw):
        # compute per-pixel loss
        ((loss1, msk1), (loss2, msk2), (loss3, msk3), (loss4, msk4)), details = self.pixel_loss(input_dict, output_dict, **kw)
        if loss1.numel() == 0:
            print('NO VALID POINTS in img1', flush=True)
        if loss2.numel() == 0:
            print('NO VALID POINTS in flow1', flush=True)
        if loss3.numel() == 0:
            print('NO VALID POINTS in img2', flush=True)
        if loss4.numel() == 0:
            print('NO VALID POINTS in flow2', flush=True)

        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(output_dict['conf_1'][msk1])
        conf2, log_conf2 = self.get_conf_log(output_dict['conf_flow_1'][msk2])
        conf3, log_conf3 = self.get_conf_log(output_dict['conf_2'][msk3])
        conf4, log_conf4 = self.get_conf_log(output_dict['conf_flow_2'][msk4])
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2
        conf_loss3 = loss3 * conf3 - self.alpha * log_conf3
        conf_loss4 = loss4 * conf4 - self.alpha * log_conf4

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
        conf_loss3 = conf_loss3.mean() if conf_loss3.numel() > 0 else 0
        conf_loss4 = conf_loss4.mean() if conf_loss4.numel() > 0 else 0

        return conf_loss1 + conf_loss2 + conf_loss3 + conf_loss4, dict(
            conf_loss_flow_1=float(conf_loss1), 
            conf_loss_flow_2=float(conf_loss2), 
            conf_loss_flow_3=float(conf_loss3), 
            conf_loss_flow_4=float(conf_loss4), 
            **details)


class ConfLoss_flow(MultiLoss):
    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction('none')
    
    def get_name(self):
        return f'ConfLoss_flow({self.pixel_loss})'

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, input_dict, output_dict, **kw):
        # compute per-pixel loss
        ((loss1, msk1), (loss2, msk2)), details = self.pixel_loss(input_dict, output_dict, **kw)
        if loss1.numel() == 0:
            print('NO VALID POINTS in img1', flush=True)
        if loss2.numel() == 0:
            print('NO VALID POINTS in img2', flush=True)

        # weight by confidence
        conf1, log_conf1 = self.get_conf_log(output_dict['conf_flow_1'][msk1])
        conf2, log_conf2 = self.get_conf_log(output_dict['conf_flow_2'][msk2])
        conf_loss1 = loss1 * conf1 - self.alpha * log_conf1
        conf_loss2 = loss2 * conf2 - self.alpha * log_conf2

        # average + nan protection (in case of no valid pixels at all)
        conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
        conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0

        return conf_loss1 + conf_loss2, dict(conf_loss_flow_1=float(conf_loss1), conf_loss_flow_2=float(conf_loss2), **details)
