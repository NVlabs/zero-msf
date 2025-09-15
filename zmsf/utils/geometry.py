# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/


import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import zmsf.utils.path_to_mast3r
import mast3r.utils.path_to_dust3r
from dust3r.utils.misc import invalid_to_zeros, invalid_to_nans
from dust3r.utils.geometry import normalize_pointcloud as normalize_pointcloud_dust3r


def visualize_camera_rays(pixels, pts3d, K, save_dir="test_vis.png", sample_rate=50):
    """
    Visualize 3D lines connecting camera center, pixel centers, and 3D points.
    
    Args:
    pixels: torch.Tensor of shape (B, H, W, 2)
    pts3d: torch.Tensor of shape (B, H, W, 3)
    K: torch.Tensor of shape (3, 3), camera intrinsic matrix
    sample_rate: int, sample every nth pixel to reduce clutter
    """
    # Ensure we're working with the first batch for simplicity
    pixels = pixels[0].cpu().numpy()
    pts3d = pts3d[0].cpu().numpy()
    K = K.cpu().numpy()
    
    # Create a 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Camera center (origin)
    camera_center = np.array([0, 0, 0])
    
    # Function to project pixel to 3D point on image plane
    def pixel_to_3d(pixel):
        x, y = pixel
        z = 1.0  # Set z to 1 (image plane)
        point_3d = np.linalg.inv(K) @ np.array([x, y, 1])
        return point_3d / point_3d[2]  # Normalize to ensure z=1
    
    # Sample pixels and 3D points
    H, W = pixels.shape[:2]
    for i in range(0, H, sample_rate):
        for j in range(0, W, sample_rate):
            pixel = pixels[i, j]
            point3d = pts3d[i, j]
            pixel[0] += W/2
            pixel[1] += H/2
            # Project pixel to 3D point on image plane
            pixel_3d = pixel_to_3d(pixel)
            
            # Line from camera center to pixel on image plane
            ax.plot([camera_center[0], pixel_3d[0]], 
                    [camera_center[1], pixel_3d[1]], 
                    [camera_center[2], pixel_3d[2]], 'r-', alpha=0.1)
            
            # Line from pixel on image plane to 3D point
            ax.plot([pixel_3d[0], point3d[0]], 
                    [pixel_3d[1], point3d[1]], 
                    [pixel_3d[2], point3d[2]], 'b-', alpha=0.1)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Rays Visualization')
    
    # Show the plot
    plt.savefig(save_dir)
    plt.close()


def crop_image_boundaries(images, crop_boundary):
    """
    Crop the boundaries of batched images by a specified ratio.
    
    Args:
    images (torch.Tensor): Batched images with shape Bx3xHxW
    crop_boundary (float): Ratio of height/width to crop from each side (0 to 0.5)
    
    Returns:
    torch.Tensor: Cropped images with shape Bx3x(H*(1-2*crop_boundary))x(W*(1-2*crop_boundary))
    """
    if not 0 <= crop_boundary < 0.5:
        raise ValueError("crop_boundary must be between 0 and 0.5")

    B, C, H, W = images.shape
    
    # Calculate the number of pixels to crop from each side
    crop_h = int(H * crop_boundary)
    crop_w = int(W * crop_boundary)
    
    # Directly crop the images using slicing
    cropped_images = images[:, :, crop_h:H-crop_h, crop_w:W-crop_w]
    
    return cropped_images


def project_point_cloud_to_image_batch(point_cloud, images, intrinsics):
    """
    Project 3D point cloud to 2D image plane for a batch of images with corresponding intrinsics.
    
    Args:
    point_cloud (torch.Tensor): Point cloud with shape BxHxWx3
    images (torch.Tensor): Original images with shape Bx3xHxW
    intrinsics (torch.Tensor): Camera intrinsics with shape Bx3x3
    
    Returns:
    torch.Tensor: Projected image with shape Bx3xHxW
    """
    B, H, W, _ = point_cloud.shape
    
    # Reshape point cloud to BxHWx3
    points = point_cloud.view(B, H*W, 3)
     
    # Project points using camera intrinsics
    proj = torch.bmm(points, intrinsics.transpose(1, 2))  # BxHWx3
    
    # Normalize by z coordinate
    proj_normalized = proj[..., :2] / (proj[..., 2:] + 1e-7)  # BxHWx2
    
    # The projection now gives coordinates centered around (0,0)
    # Scale to image coordinates
    
    # Reshape to BxHxWx2
    grid = proj_normalized.view(B, H, W, 2)

    # Normalize grid coordinates to [-1, 1] range for grid_sample
    grid[..., 0] = (grid[..., 0] / W) * 2 - 1
    grid[..., 1] = (grid[..., 1] / H) * 2 - 1

    # Sample from original images
    projected_images = F.grid_sample(images, grid, align_corners=True)
    
    return projected_images  # Bx3xHxW


def first_flow_to_optical_flow(
    flow_3d_fwd, flow_3d_bwd,
    points_src, points_tgt,
    intrinsics_src, intrinsics_tgt,
    camera_pose_src, camera_pose_tgt, eps=1e-8):
    """
    Convert first camera space 3D flow to optical flow, given 3D points instead of depth.

    Args:
    flow_3d_fwd/flow_3d_bwd (torch.Tensor): First camera space 3D flow, shape (B, H, W, 3)
    points_src/points_tgt (torch.Tensor): 3D points in first camera space, shape (B, H, W, 3)
    intrinsics_src/intrinsics_tgt (torch.Tensor): Source and Target camera intrinsic parameters, shape (B, 3, 3)
    camera_pose_src (torch.Tensor): Source camera pose (world to camera), shape (B, 4, 4)
    camera_pose_tgt (torch.Tensor): Target camera pose (world to camera), shape (B, 4, 4)
    eps (float): Small value to avoid division by zero

    Returns:
    torch.Tensor: a list of Optical flow, shape (B, H, W, 2)
    """
    B, H, W, _ = flow_3d_fwd.shape

    # Convert camera coordinates to left camera coordinates
    
    # Apply first space flow to get first space point clouds
    points_src_moved = points_src + flow_3d_fwd
    points_tgt_moved = points_tgt + flow_3d_bwd

    # Convert left camera coordinates back to right camera coordinates (in target camera frame)
    points_src_moved_right = world_to_camera_coordinates(
        camera_to_world_coordinates(points_src_moved, camera_pose_src),
        camera_pose_tgt)
    points_tgt_right = world_to_camera_coordinates(
        camera_to_world_coordinates(points_tgt, camera_pose_src),
        camera_pose_tgt)

    # Compute camera space flow
    flow_3d_fwd_cam = points_src_moved_right - points_src
    flow_3d_bwd_cam = points_tgt_moved - points_tgt_right 

    # Use the provided function to convert camera space flow to optical flow
    optical_flow_fwd = scene_flow_to_optical_flow(
        flow_3d_fwd_cam, 
        intrinsics_src, points_src)
    optical_flow_bwd = scene_flow_to_optical_flow(
        flow_3d_bwd_cam,
        intrinsics_tgt, points_tgt_right
    )

    return optical_flow_fwd, optical_flow_bwd


def world_flow_to_optical_flow(flow_3d_fwd_world, points_3d, intrinsics_src, camera_pose_src, camera_pose_tgt, eps=1e-8):
    """
    Convert world space 3D flow to optical flow, given 3D points instead of depth.

    Args:
    flow_3d_fwd_world (torch.Tensor): World space 3D flow, shape (B, H, W, 3)
    points_3d (torch.Tensor): 3D points in source camera space, shape (B, H, W, 3)
    intrinsics_src (torch.Tensor): Source camera intrinsic parameters, shape (B, 3, 3)
    camera_pose_src (torch.Tensor): Source camera pose (world to camera), shape (B, 4, 4)
    camera_pose_tgt (torch.Tensor): Target camera pose (world to camera), shape (B, 4, 4)
    eps (float): Small value to avoid division by zero

    Returns:
    torch.Tensor: Optical flow, shape (B, H, W, 2)
    """
    B, H, W, _ = flow_3d_fwd_world.shape

    # Convert camera coordinates to world coordinates
    points_3d_world = camera_to_world_coordinates(points_3d, camera_pose_src)

    # Apply world space flow
    points_3d_world_moved = points_3d_world + flow_3d_fwd_world

    # Convert world coordinates back to camera coordinates (in target camera frame)
    points_3d_moved = world_to_camera_coordinates(points_3d_world_moved, camera_pose_tgt)

    # Compute camera space flow
    flow_3d_fwd_cam = points_3d_moved - points_3d

    # Use the provided function to convert camera space flow to optical flow
    optical_flow = scene_flow_to_optical_flow(
        flow_3d_fwd_cam, 
        intrinsics_src, points_3d)

    return optical_flow


def scene_flow_to_optical_flow(flow_3d_fwd, intrinsics, points_3d, eps=1e-8):
    """
    Convert camera space scene flow to optical flow.

    Args:
    flow_3d_fwd (torch.Tensor): Camera space scene flow, shape (B, H, W, 3)
    intrinsics (torch.Tensor): Camera intrinsic parameters, shape (B, 3, 3)
    points_3d (torch.Tensor): Unprojected point map in camera space, shape (B, H, W, 3)
    eps (float): Small value to avoid division by zero

    Returns:
    torch.Tensor: Optical flow, shape (B, H, W, 2)
    """
    B, H, W, _ = flow_3d_fwd.shape

    # Compute moved 3D points
    points_3d_moved = points_3d + flow_3d_fwd  # (B, H, W, 3)

    # Project both original and moved points to 2D
    points_2d = torch.matmul(points_3d.view(B, H*W, 3), intrinsics.transpose(-1, -2)).view(B, H, W, 3)  # (B, H, W, 3)
    points_2d_moved = torch.matmul(points_3d_moved.view(B, H*W, 3), intrinsics.transpose(-1, -2)).view(B, H, W, 3)  # (B, H, W, 3)

    # Normalize homogeneous coordinates
    points_2d = points_2d[..., :2] / (points_2d[..., 2:] + eps)
    points_2d_moved = points_2d_moved[..., :2] / (points_2d_moved[..., 2:] + eps)

    # Compute optical flow
    optical_flow = points_2d_moved - points_2d  # (B, H, W, 2)

    return optical_flow


def normalize_pointcloud(pts1, pts2, norm_mode='avg_dis', valid1=None, valid2=None, ret_factor=False, norm_factor=None):
    """ renorm pointmaps pts1, pts2 with norm_mode
    """
    assert pts1.ndim >= 3 and pts1.shape[-1] == 3
    assert pts2 is None or (pts2.ndim >= 3 and pts2.shape[-1] == 3)
    
    if norm_factor is None:
        return normalize_pointcloud_dust3r(pts1, pts2, norm_mode, valid1, valid2, ret_factor)
    
    assert norm_factor.ndim == pts1.ndim

    res = pts1 / norm_factor
    if pts2 is not None:
        res = (res, pts2 / norm_factor)
    if ret_factor:
        if isinstance(res, tuple):
            res = res + (norm_factor,)
        else:
            res = (res, norm_factor,)
    return res


def compose_scene_flow(
    pts3d_left,
    flow_left_to_right,
    depth_change,
    intrinsics,
):
    """
    Compose Camera Space Scene Flow with Left camera space 3D Points, optical flow and depth change.
    
    Args:
    - pts3d_left: BxHxWx3
    - flow_left_to_right: BxHxWx2 in pixel space
    - depth_change: BxHxWx1
    - intrinsics: torch.Tensor of shape (B, 3, 3) containing camera intrinsics
    """
    B, H, W, _ = pts3d_left.shape
    device = pts3d_left.device

    # Create a grid of pixel coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), 
                                        torch.arange(W, device=device),
                                        indexing='ij')
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float()  # HxWx2
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1, -1)  # BxHxWx2

    # Add flow to get corresponding pixels in the right image
    right_pixels = pixel_coords + flow_left_to_right
    right_depth = (pts3d_left[..., -1:] + depth_change).squeeze(-1)

    pts3d_flowed = pixel_to_camera_coordinates(
        pixels=right_pixels, 
        depth=right_depth, 
        intrinsics=intrinsics)
    
    return pts3d_flowed - pts3d_left


def compute_scene_flow(
    pts3d_left, 
    flow_left_to_right, 
    pts3d_right):
    """
    Compute scene flow in the left camera space.
    
    Args:
    pts3d_left: Left camera's 3D points (BxHxWx3)
    flow_left_to_right: Optical flow from left to right image (BxHxWx2)
    pts3d_right: Right camera's 3D points in left camera space (BxHxWx3)
    
    Returns:
    scene_flow: Scene flow for each point in pts3d_left (BxHxWx3)
    """
    B, H, W, _ = pts3d_left.shape
    device = pts3d_left.device

    # Create a grid of pixel coordinates
    y_coords, x_coords = torch.meshgrid(torch.arange(H, device=device), 
                                        torch.arange(W, device=device),
                                        indexing='ij')
    pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float()  # HxWx2
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1, -1)  # BxHxWx2

    # Add flow to get corresponding pixels in the right image
    right_pixels = pixel_coords + flow_left_to_right

    # Normalize coordinates to [-1, 1] for grid_sample
    right_pixels_normalized = 2.0 * right_pixels / torch.tensor([W-1, H-1], device=device) - 1.0

    # Sample 3D points from right image using the flow
    right_pts3d_sampled = torch.nn.functional.grid_sample(pts3d_right.permute(0, 3, 1, 2), 
                                        right_pixels_normalized,
                                        mode='bilinear', 
                                        padding_mode='border', 
                                        align_corners=True)
    right_pts3d_sampled = right_pts3d_sampled.permute(0, 2, 3, 1)

    # Compute scene flow
    scene_flow = right_pts3d_sampled - pts3d_left

    return scene_flow


def RGB2SH(rgb, C0 = 0.28209479177387814):
    return (rgb - 0.5) / C0


def camera_coordinates_to_pixel_space(X_cam, camera_intrinsics):
    """
    Args:
        - X_cam (BxHxWx3 array): Camera space coordinates
        - camera_intrinsics: a Bx3x3 matrix
    Returns:
        u, v coordinates in pixel space (BxHxW arrays)
    """
    B, H, W, _ = X_cam.shape

    # Extract camera intrinsics
    fu = camera_intrinsics[:, 0:1, 0:1]
    fv = camera_intrinsics[:, 1:2, 1:2]
    cu = camera_intrinsics[:, 0:1, 2:3]
    cv = camera_intrinsics[:, 1:2, 2:3]

    # Extract x, y, z components
    x_cam = X_cam[..., 0]
    y_cam = X_cam[..., 1]
    z_cam = X_cam[..., 2]
    
    # Project back to pixel space
    u = (x_cam * fu / z_cam) + cu # BxHxW
    v = (y_cam * fv / z_cam) + cv # BxHxW

    return torch.stack([u, v], dim=-1)


def pixel_to_camera_coordinates(pixels, depth, intrinsics):
    """
    Lift pixels and depth to 3D points in camera coordinates.
    
    Args:
    - pixels: torch.Tensor of shape (B, H, W, 2) containing pixel coordinates
    - depth: torch.Tensor of shape (B, H, W) containing depth values
    - intrinsics: torch.Tensor of shape (B, 3, 3) containing camera intrinsics
    
    Returns:
    - X_cam: torch.Tensor of shape (B, H, W, 3) containing 3D points in camera coordinates
    """
    B, H, W, _ = pixels.shape

    # Extract intrinsic parameters
    fx = intrinsics[:, 0, 0].view(B, 1, 1)
    fy = intrinsics[:, 1, 1].view(B, 1, 1)
    cx = intrinsics[:, 0, 2].view(B, 1, 1)
    cy = intrinsics[:, 1, 2].view(B, 1, 1)

    # Extract pixel coordinates
    x = pixels[..., 0]
    y = pixels[..., 1]

    # Compute normalized coordinates
    x_norm = (x - cx) / fx
    y_norm = (y - cy) / fy

    # Compute 3D coordinates
    X = x_norm * depth
    Y = y_norm * depth
    Z = depth

    # Stack to create 3D points
    X_cam = torch.stack([X, Y, Z], dim=-1)

    return X_cam


def world_to_camera_vector(V_world, camera_pose):
    """
    Convert world vectors to camera vectors (apply rotation only).
    
    Args:
    - V_world: torch.Tensor of shape (B, H, W, 3) containing 3D vectors in world coordinates
    - camera_pose: torch.Tensor of shape (B, 4, 4) representing the camera poses
    
    Returns:
    - V_cam: torch.Tensor of shape (B, H, W, 3) containing 3D vectors in camera coordinates
    """
    B, H, W, _ = V_world.shape
    R_world2cam = camera_pose[:, :3, :3]
    
    V_world_reshaped = V_world.view(B, H*W, 3)
    V_cam = torch.bmm(V_world_reshaped, R_world2cam.transpose(1, 2))
    V_cam = V_cam.view(B, H, W, 3)
    
    return V_cam


def camera_to_world_vector(V_camera, camera_pose):
    """
    Convert camera vectors to world vectors (apply inverse rotation).
    
    Args:
    - V_camera: torch.Tensor of shape (B, H, W, 3) containing 3D vectors in camera coordinates
    - camera_pose: torch.Tensor of shape (B, 4, 4) representing the camera poses
    
    Returns:
    - V_world: torch.Tensor of shape (B, H, W, 3) containing 3D vectors in world coordinates
    """
    B, H, W, _ = V_camera.shape
    R_world2cam = camera_pose[:, :3, :3]
    
    # The inverse rotation is just the transpose for orthogonal matrices
    R_cam2world = R_world2cam.transpose(1, 2)
    
    V_camera_reshaped = V_camera.view(B, H*W, 3)
    V_world = torch.bmm(V_camera_reshaped, R_cam2world)
    V_world = V_world.view(B, H, W, 3)
    
    return V_world


def world_to_camera_coordinates(X_world, camera_pose):
    """
    Convert world coordinates to camera coordinates.
    
    Args:
    - X_world: torch.Tensor of shape (B, H, W, 3) or (B, N, 3) containing 3D points in world coordinates
    - camera_pose: torch.Tensor of shape (B, 4, 4) representing the camera poses (world to camera transformation)
    
    Returns:
    - X_cam: torch.Tensor of shape (B, H, W, 3) or (B, N, 3) containing 3D points in camera coordinates
    """
    if len(X_world.shape) == 4:
        B, H, W, _ = X_world.shape
        reshape_needed = True
    elif len(X_world.shape) == 3:
        B, N, _ = X_world.shape
        reshape_needed = False
    else:
        raise ValueError("X_world must have shape (B, H, W, 3) or (B, N, 3)")

    # Extract rotation and translation from camera pose
    R_world2cam = camera_pose[:, :3, :3]
    t_cam2world = camera_pose[:, :3, 3]

    if reshape_needed:
        # Reshape X_world to (B, H*W, 3)
        X_world_reshaped = X_world.view(B, H*W, 3)
    else:
        X_world_reshaped = X_world

    # Transform points
    X_cam = torch.bmm(X_world_reshaped - t_cam2world.unsqueeze(1), R_world2cam)

    if reshape_needed:
        # Reshape back to (B, H, W, 3)
        X_cam = X_cam.view(B, H, W, 3)

    return X_cam


def world_to_camera_coordinates_old(X_world, camera_pose):
    """
    Convert world coordinates to camera coordinates.
    
    Args:
    - X_world: torch.Tensor of shape (B, H, W, 3) containing 3D points in world coordinates
    - camera_pose: torch.Tensor of shape (B, 4, 4) representing the camera poses (world to camera transformation)
    
    Returns:
    - X_cam: torch.Tensor of shape (B, H, W, 3) containing 3D points in camera coordinates
    """
        
    B, H, W, _ = X_world.shape  
    # Extract rotation and translation from camera pose
    R_world2cam = camera_pose[:, :3, :3]
    t_cam2world = camera_pose[:, :3, 3]

    # Reshape X_world to (B, H*W, 3)
    X_world_reshaped = X_world.view(B, H*W, 3)

    # Transform points
    X_cam = torch.bmm(X_world_reshaped - t_cam2world.unsqueeze(1), R_world2cam)

    # Reshape back to (B, H, W, 3)
    X_cam = X_cam.view(B, H, W, 3)

    return X_cam


def camera_to_world_coordinates(X_cam, camera_pose):
    """
    Convert camera coordinates to world coordinates.
    
    Args:
    - X_cam: torch.Tensor of shape (B, H, W, 3) or (B, N, 3) containing 3D points in camera coordinates
    - camera_pose: torch.Tensor of shape (B, 4, 4) representing the camera poses (world to camera transformation)
    
    Returns:
    - X_world: torch.Tensor of shape (B, H, W, 3) or (B, N, 3) containing 3D points in world coordinates
    """
    if len(X_cam.shape) == 4:
        B, H, W, _ = X_cam.shape
        reshape_needed = True
    elif len(X_cam.shape) == 3:
        B, N, _ = X_cam.shape
        reshape_needed = False
    else:
        raise ValueError("X_cam must have shape (B, H, W, 3) or (B, N, 3)")

    # Extract rotation and translation from camera pose
    R_world2cam = camera_pose[:, :3, :3]
    t_cam2world = camera_pose[:, :3, 3]

    # Compute inverse transformation
    R_cam2world = R_world2cam.transpose(1, 2).contiguous()  # Shape: (B, 3, 3)
    
    if reshape_needed:
        # Reshape X_cam to (B, H*W, 3)
        X_cam_reshaped = X_cam.view(B, H*W, 3)
    else:
        X_cam_reshaped = X_cam

    # Transform points
    X_world = torch.bmm(X_cam_reshaped, R_cam2world) + t_cam2world.unsqueeze(1)

    if reshape_needed:
        # Reshape back to (B, H, W, 3)
        X_world = X_world.view(B, H, W, 3)

    return X_world


def camera_to_world_coordinates_old(X_cam, camera_pose):
    """
    Convert camera coordinates to world coordinates.
    
    Args:
    - X_cam: torch.Tensor of shape (B, H, W, 3) containing 3D points in camera coordinates
    - camera_pose: torch.Tensor of shape (B, 4, 4) representing the camera poses (world to camera transformation)
    
    Returns:
    - X_world: torch.Tensor of shape (B, H, W, 3) containing 3D points in world coordinates
    """
    B, H, W, _ = X_cam.shape

    # Extract rotation and translation from camera pose
    R_world2cam = camera_pose[:, :3, :3]
    t_cam2world = camera_pose[:, :3, 3]

    # Compute inverse transformation
    R_cam2world = R_world2cam.transpose(1, 2).contiguous()  # Shape: (B, 3, 3)
    
   
    # Reshape X_cam to (B, H*W, 3)
    X_cam_reshaped = X_cam.view(B, H*W, 3)

    # Transform points
    X_world = torch.bmm(X_cam_reshaped, R_cam2world) + t_cam2world.unsqueeze(1)

    # Reshape back to (B, H, W, 3)
    X_world = X_world.view(B, H, W, 3)

    return X_world


def align_pts_to_rays(
    pts3d,
    K):
    # input of shape: BxHxWx3
    B, H, W, _ = pts3d.shape
    x = torch.linspace(0, W, W, device=pts3d.device)
    y = torch.linspace(0, H, H, device=pts3d.device)
    
    # Create meshgrid
    y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
    
    # Stack x and y coordinates
    pp = K[:, :2, 2:] # Bx2x1
    #assert False, pp
    pixels = torch.stack([x_grid, y_grid], dim=-1)[None].repeat(B, 1, 1, 1).view(B, -1, 2) #- pp.view(B, 1, 2) # Bx(HxW)x2
    
    # Function to project pixel to 3D point on image plane
    # return: Bx3x(HxW)
    def pixel_to_3d(pixels):
        # pixels: Bx(HxW)x2
        xyz = torch.cat([pixels, torch.ones(B, H*W, 1, device=pixels.device, dtype=pixels.dtype)], dim=-1).permute(0, 2, 1) # Bx3x(HxW)
        pixels_3d = torch.linalg.inv(K) @ xyz # Bx3x(HxW)
        return torch.nn.functional.normalize(pixels_3d, dim=1) # normalize to get camera ray direction
    
    pixels_3d = pixel_to_3d(pixels) # Bx3x(HxW), ray directions
    pixels_3d = pixels_3d.permute(0, 2, 1).view(B, H, W, 3)

    # Calculate the projection of pts3d onto the ray directions
    projections = torch.sum(pts3d * pixels_3d, dim=-1, keepdim=True)
    
    # Calculate the closest points on the rays
    closest_points = projections * pixels_3d
    return closest_points


def render_translate_pose(
    reconstruction_right,
    world_to_cam1,
    world_to_cam2,
):
    #### Gaussian for right -> right
    reconstruction_right_right = {key: value for key,value in reconstruction_right.items()}
    ### Translate xyz
    # Ensure points are on the same device as extrinsics
    point_cloud = reconstruction_right_right["xyz"][None]
    world_to_cam1 = world_to_cam1[None]
    world_to_cam2 = world_to_cam2[None]
    
    # Camera 1 space to world space
    R_world2cam1 = world_to_cam1[:, :3, :3]
    t_cam12world = world_to_cam1[:, :3, 3]
    R_cam12world = R_world2cam1.transpose(1, 2).contiguous()  # Shape: (B, 3, 3)
    point_cloud = torch.bmm(point_cloud, R_cam12world) + t_cam12world.unsqueeze(1)

    # world space to camera 2 space
    R_world2cam2 = world_to_cam2[:, :3, :3]
    t_cam22world = world_to_cam2[:, :3, 3]
    point_cloud = torch.bmm(point_cloud - t_cam22world.unsqueeze(1), R_world2cam2)#.transpose(1, 2).contiguous()) + t_world2cam2.unsqueeze(1)

    # set point cloud
    reconstruction_right_right["xyz"] = point_cloud[0]
    world_to_cam1 = world_to_cam1[0]
    world_to_cam2 = world_to_cam2[0]
 
    ### Translate scale
    # Assuming R is the rotation matrix from cam1 to cam2
    R = torch.matmul(world_to_cam2[:3, :3], torch.inverse(world_to_cam1[:3, :3]))
    # For each Gaussian
    covariance_cam1 = torch.diag_embed(reconstruction_right_right["scaling"]**2)  # Convert scale to covariance
    covariance_cam2 = R @ covariance_cam1 @ R.T
    # New scale is the square root of the diagonal of the new covariance
    reconstruction_right_right["scaling"] = torch.sqrt(torch.diagonal(covariance_cam2, dim1=-2, dim2=-1))
    ### Translate rotation
    # Assuming rotation_cam1 is in quaternion format
    rotation_cam1_matrix = quaternion_to_rotation_matrix(reconstruction_right_right["rotation"])
    # Compose rotations
    rotation_cam2_matrix = R @ rotation_cam1_matrix
    # Convert back to quaternion if needed
    reconstruction_right_right["rotation"] = rotation_matrix_to_quaternion(rotation_cam2_matrix)
    
    return reconstruction_right_right


def quaternion_to_rotation_matrix(quaternions):
    """
    Convert quaternions to rotation matrices.
    
    Args:
    quaternions (torch.Tensor): Quaternions of shape (..., 4)
    
    Returns:
    torch.Tensor: Rotation matrices of shape (..., 3, 3)
    """
    # Normalize quaternions
    quaternions = F.normalize(quaternions, dim=-1)
    
    # Extract components
    w, x, y, z = torch.unbind(quaternions, -1)
    
    # Compute terms
    xy = x * y
    xz = x * z
    yz = y * z
    xw = x * w
    yw = y * w
    zw = z * w
    xx = x * x
    yy = y * y
    zz = z * z
    
    # Construct rotation matrix
    rotation_matrix = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)
    ], dim=-1).reshape(quaternions.shape[:-1] + (3, 3))
    
    return rotation_matrix


def rotation_matrix_to_quaternion(rotation_matrix):
    """
    Convert rotation matrices to quaternions.
    
    Args:
    rotation_matrix (torch.Tensor): Rotation matrices of shape (..., 3, 3)
    
    Returns:
    torch.Tensor: Quaternions of shape (..., 4)
    """
    if rotation_matrix.shape[-1] != 3 or rotation_matrix.shape[-2] != 3:
        raise ValueError("Invalid rotation matrix shape")
    
    batch_dim = rotation_matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(rotation_matrix.reshape(*batch_dim, 9), dim=-1)
    
    q_abs = torch.stack([
        torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0)) / 2,
        torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0)) / 2,
        torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0)) / 2,
        torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0)) / 2,
    ], dim=-1)
    
    max_indices = torch.argmax(q_abs, dim=-1)
    signs = torch.stack([
        torch.sign(m21 - m12),
        torch.sign(m02 - m20),
        torch.sign(m10 - m01),
        torch.ones_like(m00)
    ], dim=-1)
    
    q = torch.zeros_like(q_abs)
    q.scatter_(dim=-1, index=max_indices.unsqueeze(-1), src=q_abs.gather(-1, max_indices.unsqueeze(-1)))
    q = q * signs
    
    q = torch.stack([q[..., 3], q[..., 0], q[..., 1], q[..., 2]], dim=-1)
    return F.normalize(q, dim=-1)


def create_colored_image(points_2d, colors, height, width):
    """Create an image from colored 2D points."""
    image = torch.ones((3, height, width), device=points_2d.device)
    
    points_2d = points_2d.long()
    
    # Filter points within image bounds
    mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < width) & (points_2d[:, 1] >= 0) & (points_2d[:, 1] < height)
    
    
    points_2d = points_2d[mask]
    colors = colors[mask]
    
    # Assign colors to image
    image[:, points_2d[:, 1], points_2d[:, 0]] = colors.T
    
    return image
