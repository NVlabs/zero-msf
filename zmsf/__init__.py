# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from .model import AsymmetricMASt3R_flow
from .utils.geometry import world_to_camera_coordinates, pixel_to_camera_coordinates, camera_to_world_coordinates, world_to_camera_vector


def compute_cycle_consistency(
    flow_fwd, flow_bwd, 
    depth_src, depth_tgt, 
    rgb_src, rgb_tgt, 
    intrinsics_src, intrinsics_tgt,
    camera_pose_src, camera_pose_tgt,
    alpha=0.01, beta=0.5, diff_threshold=0.5,
    camera_space=False,
    first_camera=False,
    visualize=False,
    depth_threshold=0.): 
    """
    Compute cycle consistency masks for forward and backward optical flow,
    considering depth information for occlusion handling.
    
    Args:
    flow_fwd (torch.Tensor): Forward flow of shape (B, H, W, 2)
    flow_bwd (torch.Tensor): Backward flow of shape (B, H, W, 2)
    depth_src (torch.Tensor): Source depth map of shape (B, 1, H, W)
    depth_tgt (torch.Tensor): Target depth map of shape (B, 1, H, W)
    rgb_src (torch.Tensor, optional): Source RGB image of shape (B, 3, H, W)
    rgb_tgt (torch.Tensor, optional): Target RGB image of shape (B, 3, H, W)
    alpha (float): Threshold for cycle consistency
    beta (float): Scaling factor for the cycle consistency check
    
    Returns:
    Tuple[torch.Tensor, torch.Tensor]: Forward mask, backward mask
    """

    assert visualize is False, "visualize is not supported"

    B, H, W, _ = flow_fwd.shape
    
    # Create coordinate grid
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords = torch.stack((x_coords, y_coords)).float().to(flow_fwd.device)
    coords = coords[None].repeat(B, 1, 1, 1)  # Shape: (B, 2, H, W)
    coords = coords.permute(0, 2, 3, 1)

    # Warp coordinates with forward and backward flows
    coords_fwd = coords + flow_fwd
    coords_bwd = coords + flow_bwd

    # Normalize coordinates to [-1, 1] range for grid_sample
    coords_fwd_norm = coords_fwd / torch.tensor([W - 1, H - 1]).to(flow_fwd.device) * 2 - 1
    coords_bwd_norm = coords_bwd / torch.tensor([W - 1, H - 1]).to(flow_bwd.device) * 2 - 1


    # Sample flows at warped coordinates
    flow_bwd_warped = F.grid_sample(flow_bwd.permute(0, 3, 1, 2), coords_fwd_norm, align_corners=True).permute(0, 2, 3, 1)
    flow_fwd_warped = F.grid_sample(flow_fwd.permute(0, 3, 1, 2), coords_bwd_norm, align_corners=True).permute(0, 2, 3, 1)

    # Compute cycle consistency errors
    cycle_diff_fwd = flow_fwd + flow_bwd_warped
    cycle_diff_bwd = flow_bwd + flow_fwd_warped
    
    cycle_dist_fwd = torch.norm(cycle_diff_fwd, dim=-1)
    cycle_dist_bwd = torch.norm(cycle_diff_bwd, dim=-1)

    # consistency check inference:https://github.com/haofeixu/gmflow/blob/b5123431164d01ec14526a1c3d22218aecb62024/gmflow/geometry.py#L75
    # Compute flow magnitudes
    flow_mag_fwd = torch.norm(flow_fwd, dim=-1)
    flow_mag_bwd = torch.norm(flow_bwd, dim=-1)
    flow_mag = flow_mag_fwd + flow_mag_bwd # [B, H, W]
    # Compute masks
    consistency_mask_fwd = cycle_dist_fwd < (alpha * flow_mag + beta)
    consistency_mask_bwd = cycle_dist_bwd < (alpha * flow_mag + beta)
   
    # Ensure RGB images are normalized
    rgb_src_norm = rgb_src 
    rgb_tgt_norm = rgb_tgt

    # Warp images
    rgb_src_warped = F.grid_sample(rgb_tgt_norm, coords_fwd_norm, align_corners=True)
    rgb_tgt_warped = F.grid_sample(rgb_src_norm, coords_bwd_norm, align_corners=True)
    
    if diff_threshold is not None:
        # Compute raw differences between original and warped images
        diff_src = torch.abs(rgb_src_norm - rgb_src_warped).mean(dim=1) < diff_threshold
        diff_tgt = torch.abs(rgb_tgt_norm - rgb_tgt_warped).mean(dim=1) < diff_threshold
    else:
        diff_src = torch.abs(rgb_src_norm - rgb_src_warped).mean(dim=1) >= 0.
        diff_tgt = torch.abs(rgb_tgt_norm - rgb_tgt_warped).mean(dim=1) >= 0.
    
    # Combine consistency and occlusion masks
    # has to be both true to be true

    # Warp depth maps
    # all depth are >= 0
    # for every pixel of the left image, the gt depth after offset
    depth_src_warped = F.grid_sample(depth_tgt, coords_fwd_norm, align_corners=True).squeeze(1)
    # for every pixel of the right image, the gt depth after offset
    depth_tgt_warped = F.grid_sample(depth_src, coords_bwd_norm, align_corners=True).squeeze(1)

    # Create depth validity masks
    valid_depth_src = depth_src.squeeze(1) > depth_threshold
    valid_depth_tgt = depth_tgt.squeeze(1) > depth_threshold
    valid_depth_src_warped = depth_src_warped > depth_threshold
    valid_depth_tgt_warped = depth_tgt_warped > depth_threshold

    # Combine consistency, occlusion, and depth validity masks
    mask_fwd = (consistency_mask_fwd & diff_src & valid_depth_src & valid_depth_src_warped).float()
    mask_bwd = (consistency_mask_bwd & diff_tgt & valid_depth_tgt & valid_depth_tgt_warped).float()
    
    # get 4 set of camera space points
    X_cam_src = pixel_to_camera_coordinates(coords, depth_src.squeeze(1), intrinsics_src)
    X_cam_src_warped = pixel_to_camera_coordinates(coords_fwd, depth_src_warped, intrinsics_tgt)
    X_cam_tgt = pixel_to_camera_coordinates(coords, depth_tgt.squeeze(1), intrinsics_tgt)
    X_cam_tgt_warped = pixel_to_camera_coordinates(coords_bwd, depth_tgt_warped, intrinsics_src)

    # get 4 set of scene space points
    if camera_space:
        if first_camera:
            X_world_src = X_cam_src # already in left
            X_world_tgt_warped = X_cam_tgt_warped # already in left
            X_world_src_warped = world_to_camera_coordinates(
                camera_to_world_coordinates(X_cam_src_warped, camera_pose_tgt),
                camera_pose_src
            )
            X_world_tgt = world_to_camera_coordinates(
                camera_to_world_coordinates(X_cam_tgt, camera_pose_tgt),
                camera_pose_src
            )

        else:
            X_world_src  = X_cam_src
            X_world_src_warped = X_cam_src_warped
            X_world_tgt = X_cam_tgt
            X_world_tgt_warped = X_cam_tgt_warped
    else:
        X_world_src = camera_to_world_coordinates(X_cam_src, camera_pose_src)
        X_world_src_warped = camera_to_world_coordinates(X_cam_src_warped, camera_pose_tgt)
        X_world_tgt = camera_to_world_coordinates(X_cam_tgt, camera_pose_tgt)
        X_world_tgt_warped = camera_to_world_coordinates(X_cam_tgt_warped, camera_pose_src)

    # get fwd flow and bwd flow
    flow_3d_fwd = X_world_src_warped - X_world_src
    flow_3d_bwd = X_world_tgt_warped - X_world_tgt

    # get fwd and bwd mask
    mask_3d_fwd = mask_fwd
    mask_3d_bwd = mask_bwd

    return flow_3d_fwd, flow_3d_bwd, mask_3d_fwd.bool(), mask_3d_bwd.bool()


def compute_world_flow(
    flow_3d_fwd, flow_3d_bwd,
    depth_src, depth_tgt,
    intrinsics_src, intrinsics_tgt,
    camera_pose_src, camera_pose_tgt
    ):
    B, H, W, _ = flow_3d_fwd.shape

    # Create coordinate grid
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords = torch.stack((x_coords, y_coords)).float().to(flow_3d_fwd.device)
    coords = coords[None].repeat(B, 1, 1, 1)  # Shape: (B, 2, H, W)
    coords = coords.permute(0, 2, 3, 1)

    X_cam_src = pixel_to_camera_coordinates(coords, depth_src.squeeze(1), intrinsics_src)
    X_cam_tgt = pixel_to_camera_coordinates(coords, depth_tgt.squeeze(1), intrinsics_tgt)
    
    # this is in second camera's coordinate system now!
    X_cam_src_warped = X_cam_src + flow_3d_fwd
    # this is in first camera's coordinate system now!
    X_cam_tgt_warped = X_cam_tgt + flow_3d_bwd

    X_world_src = camera_to_world_coordinates(X_cam_src, camera_pose_src)
    X_world_src_warped = camera_to_world_coordinates(X_cam_src_warped, camera_pose_tgt)
    X_world_tgt = camera_to_world_coordinates(X_cam_tgt, camera_pose_tgt)
    X_world_tgt_warped = camera_to_world_coordinates(X_cam_tgt_warped, camera_pose_src)
   
    # get fwd flow and bwd flow
    flow_3d_fwd = X_world_src_warped - X_world_src
    flow_3d_bwd = X_world_tgt_warped - X_world_tgt

    return flow_3d_fwd, flow_3d_bwd


def compute_first_flow(
    flow_3d_fwd, flow_3d_bwd,
    depth_src, depth_tgt,
    intrinsics_src, intrinsics_tgt,
    camera_pose_src, camera_pose_tgt
    ):
    B, H, W, _ = flow_3d_fwd.shape

    # Create coordinate grid
    y_coords, x_coords = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    coords = torch.stack((x_coords, y_coords)).float().to(flow_3d_fwd.device)
    coords = coords[None].repeat(B, 1, 1, 1)  # Shape: (B, 2, H, W)
    coords = coords.permute(0, 2, 3, 1)

    X_cam_src = pixel_to_camera_coordinates(coords, depth_src.squeeze(1), intrinsics_src)
    X_cam_tgt = pixel_to_camera_coordinates(coords, depth_tgt.squeeze(1), intrinsics_tgt)
    
    # this is in second camera's coordinate system now!
    X_cam_src_warped = X_cam_src + flow_3d_fwd
    # this is in first camera's coordinate system now!
    X_cam_tgt_warped = X_cam_tgt + flow_3d_bwd

    X_left_src_warped = world_to_camera_coordinates(
        camera_to_world_coordinates(X_cam_src_warped, camera_pose_tgt),
        camera_pose_src)
    X_left_tgt = world_to_camera_coordinates(
        camera_to_world_coordinates(X_cam_tgt, camera_pose_tgt),
        camera_pose_src)
    
    # get fwd flow and bwd flow
    flow_3d_fwd = X_left_src_warped - X_cam_src
    flow_3d_bwd = X_cam_tgt_warped - X_left_tgt

    return flow_3d_fwd, flow_3d_bwd


def visualize_pcd_with_distance(gt_1, pcd_1, gt_2, pcd_2, path, subsample_factor=10, alpha=1., fig=None, ax=None):
    """
    Visualize two concatenated point clouds in a 3D scatter plot, 
    colored based on their distance to corresponding ground truth points using a single jet colormap.
    
    Args:
    - pcd_1, pcd_2: numpy arrays of shape (N, 3) containing 3D points
    - gt_1, gt_2: numpy arrays of shape (N, 3) containing corresponding ground truth points
    - subsample_factor: int, factor by which to subsample the point clouds
    - alpha: float, transparency of the points
    - fig, ax: existing figure and axis objects (optional)
    
    Returns:
    - fig, ax: the figure and axis objects used for plotting
    """
    if fig is None or ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Subsample the point clouds
    if subsample_factor > 1:
        pcd_1 = pcd_1[::subsample_factor]
        pcd_2 = pcd_2[::subsample_factor]
        gt_1 = gt_1[::subsample_factor]
        gt_2 = gt_2[::subsample_factor]
    
    # Concatenate point clouds and ground truth data
    pcd = np.vstack((pcd_1, pcd_2))
    gt = np.vstack((gt_1, gt_2))
    
    # Calculate absolute errors
    absolute_errors = np.linalg.norm(pcd - gt, axis=1)  
    # Calculate distances of ground truth points from origin
    distances = absolute_errors
    
    upper = np.quantile(distances, 0.98)
    lower = np.quantile(distances, 0.02)
    mask = (distances < upper) & (distances > lower)
    gt = gt[mask]
    distances = distances[mask]


    # Plot concatenated point cloud with jet colormap
    scatter = ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], 
                         c=distances, cmap='jet', s=1, alpha=alpha)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, aspect=40, label='Distance to GT')
    
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down)')
    ax.set_zlabel('Z (Forward)')
    ax.set_zticks([])
    # Set initial view for standard camera coordinate system
    ax.view_init(elev=90, azim=-90)
    
    # Adjust axis directions
    ax.invert_yaxis()  # Y axis points downward
    ax.invert_zaxis()
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    # Create and save PLY file using Open3D
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    
    # Create a color array using the jet colormap
    cmap = plt.get_cmap('jet')
    # Create a normalizer for the colormap
    norm = Normalize(vmin=np.min(distances), vmax=np.max(distances))
    colors = cmap(norm(distances))[:, :3]  # RGB values, drop alpha channel
    
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path.split(".")[0] +".ply", pcd_o3d)


def visualize_pcd_with_distance_flow(
    gt_1, gt_3d_fwd, flow_3d_fwd, 
    gt_2, gt_3d_bwd, flow_3d_bwd, 
    path, 
    subsample_factor=10, alpha=1., fig=None, ax=None):
    """
    Visualize two concatenated point clouds in a 3D scatter plot, 
    colored based on their distance to corresponding ground truth points using a single jet colormap.
    
    Args:
    - pcd_1, pcd_2: numpy arrays of shape (N, 3) containing 3D points
    - gt_1, gt_2: numpy arrays of shape (N, 3) containing corresponding ground truth points
    - subsample_factor: int, factor by which to subsample the point clouds
    - alpha: float, transparency of the points
    - fig, ax: existing figure and axis objects (optional)
    
    Returns:
    - fig, ax: the figure and axis objects used for plotting
    """
    if fig is None or ax is None:
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
    
    # Subsample the point clouds
    if subsample_factor > 1:
        gt_3d_fwd = gt_3d_fwd[::subsample_factor]
        gt_3d_bwd = gt_3d_bwd[::subsample_factor]
        flow_3d_fwd = flow_3d_fwd[::subsample_factor]
        flow_3d_bwd = flow_3d_bwd[::subsample_factor]
        gt_1 = gt_1[::subsample_factor]
        gt_2 = gt_2[::subsample_factor]
    
    # Concatenate point clouds and ground truth data
    gt_3d = np.vstack((gt_3d_fwd, gt_3d_bwd))
    flow_3d = np.vstack((flow_3d_fwd, flow_3d_bwd))
    gt = np.vstack((gt_1, gt_2))
    
    # Calculate absolute errors
    absolute_errors = np.linalg.norm(flow_3d - gt_3d, axis=1)  
    # Calculate distances of ground truth points from origin
    distances = absolute_errors

    upper = np.quantile(distances, 0.98)
    lower = np.quantile(distances, 0.02)
    mask = (distances < upper ) & (distances > lower)
    gt = gt[mask]
    distances = distances[mask]

    # Plot concatenated point cloud with jet colormap
    scatter = ax.scatter(gt[:, 0], gt[:, 1], gt[:, 2], 
                         c=distances, cmap='jet', s=1, alpha=alpha)
    
    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1, aspect=40, label='Distance to GT Flow')
    
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Y (Down)')
    ax.set_zlabel('Z (Forward)')
    ax.set_zticks([])
    # Set initial view for standard camera coordinate system
    ax.view_init(elev=90, azim=-90)
    
    # Adjust axis directions
    ax.invert_yaxis()  # Y axis points downward
    ax.invert_zaxis()
    
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    # Create and save PLY file using Open3D
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(gt)
    
    # Create a color array using the jet colormap
    cmap = plt.get_cmap('jet')
    # Create a normalizer for the colormap
    norm = Normalize(vmin=np.min(distances), vmax=np.max(distances))
    
    colors = cmap(norm(distances))[:, :3]  # RGB values, drop alpha channel
    
    pcd_o3d.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path.split(".")[0] +".ply", pcd_o3d)
