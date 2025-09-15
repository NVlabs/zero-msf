# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/


import argparse
import os
import yaml
import importlib
from pathlib import Path
from PIL import Image
from typing import Any, Dict

import numpy as np
import torch


def instantiate_component(config: Dict[str, Any]) -> Any:
    """Instantiate a component from config dict with class_path and init_args"""
    if 'class_path' in config:
        module_path, class_name = config['class_path'].rsplit('.', 1)
        module = importlib.import_module(module_path)
        class_ = getattr(module, class_name)
        init_args = {k: instantiate_component(v) if isinstance(v, dict) else v 
                     for k, v in config.get('init_args', {}).items()}
        return class_(**init_args)
    return {k: instantiate_component(v) if isinstance(v, dict) else v 
            for k, v in config.items()}


def load_image(image_path, target_size=(512, 384)):
    """Load image as PIL Image and resize to model's expected size"""
    img = Image.open(image_path).convert('RGB')
    orig_width, orig_height = img.size
    target_width, target_height = target_size
    
    print(f"Loading image: {orig_width}x{orig_height} -> {target_width}x{target_height}")
    
    # Resize to exact model input size (this will change aspect ratio)
    img = img.resize((target_width, target_height), Image.LANCZOS)
    
    return img, (orig_width, orig_height)


def create_mock_intrinsics(height, width, focal_length=None):
    """Create mock camera intrinsics if not provided"""
    if focal_length is None:
        # Rough estimate: focal = max(h,w) / (2 * tan(30 deg))
        focal_length = max(height, width) / 1.1547
    
    intrinsics = np.eye(3, dtype=np.float32)
    intrinsics[0, 0] = focal_length  # fx
    intrinsics[1, 1] = focal_length  # fy
    intrinsics[0, 2] = width / 2.0   # cx
    intrinsics[1, 2] = height / 2.0  # cy
    
    return intrinsics


def prepare_input_data(image1_path, image2_path, intrinsics=None, target_size=(512, 384)):
    """Prepare input data for the model"""
    # Load images and get original dimensions
    img1, (orig_width1, orig_height1) = load_image(image1_path, target_size)
    img2, (orig_width2, orig_height2) = load_image(image2_path, target_size)
    
    # Get model input dimensions
    width, height = target_size
    
    # Use the first image's original dimensions for intrinsics calculation
    orig_width, orig_height = orig_width1, orig_height1
    
    # Create or adjust intrinsics
    if intrinsics is None:
        # Create intrinsics based on original image size first
        intrinsics = create_mock_intrinsics(orig_height, orig_width)
    
    # Always adjust intrinsics for the resize to model input size
    scale_x = width / orig_width
    scale_y = height / orig_height
    
    intrinsics = intrinsics.copy()
    intrinsics[0, 0] *= scale_x  # fx
    intrinsics[1, 1] *= scale_y  # fy  
    intrinsics[0, 2] *= scale_x  # cx
    intrinsics[1, 2] *= scale_y  # cy
    
    print(f"Adjusted intrinsics: scale_x={scale_x:.3f}, scale_y={scale_y:.3f}")
    print(f"Original image aspect ratio: {orig_width/orig_height:.3f}")
    print(f"Model input aspect ratio: {width/height:.3f}")
    
    # Convert to tensors and prepare batch
    camera_pose = np.eye(4, dtype=np.float32)
    
    view1 = {
        'img': img1,
        'camera_pose': torch.from_numpy(camera_pose),
        'camera_intrinsics': torch.from_numpy(intrinsics),
        'dataset': 'demo',
        'instance': str(image1_path),
        'orig_size': (orig_width1, orig_height1)
    }
    
    view2 = {
        'img': img2, 
        'camera_pose': torch.from_numpy(camera_pose),
        'camera_intrinsics': torch.from_numpy(intrinsics),
        'dataset': 'demo',
        'instance': str(image2_path),
        'orig_size': (orig_width2, orig_height2)
    }
    
    return view1, view2


def list_to_batch(views, device):
    """Convert list of views to batched format"""
    batch = []
    for view in views:
        batched_view = {}
        for key, value in view.items():
            if isinstance(value, torch.Tensor):
                batched_view[key] = value.unsqueeze(0).to(device)
            elif isinstance(value, Image.Image):
                # Convert PIL image to tensor and normalize
                img_array = np.array(value).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
                batched_view[key] = img_tensor.to(device)
            else:
                batched_view[key] = value
        batch.append(batched_view)
    
    return batch


def save_point_cloud(points, colors, filename):
    """Save point cloud in PLY format"""
    points = points.cpu().numpy()
    colors = colors.cpu().numpy() if colors is not None else None
    
    # Reshape if needed
    if len(points.shape) == 4:  # Bx3xHxW
        B, C, H, W = points.shape
        points = points.reshape(B, C, -1).transpose(0, 2, 1).reshape(-1, C)
    elif len(points.shape) == 3:  # BxNx3
        B, N, C = points.shape 
        points = points.reshape(-1, C)
    
    if colors is not None:
        if len(colors.shape) == 4:  # Bx3xHxW
            B, C, H, W = colors.shape
            colors = colors.reshape(B, C, -1).transpose(0, 2, 1).reshape(-1, C)
        elif len(colors.shape) == 3:  # BxNx3
            B, N, C = colors.shape
            colors = colors.reshape(-1, C)
        
        # Convert to 0-255 range
        colors = (colors * 255).astype(np.uint8)
    
    # Remove invalid points (z <= 0)
    valid_mask = points[:, 2] > 0
    points = points[valid_mask]
    if colors is not None:
        colors = colors[valid_mask]
    
    print(f"Saving {len(points)} points to {filename}")
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n") 
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            if colors is not None:
                f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                       f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")
            else:
                f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f}\n")


def save_scene_flow(points, flow, filename):
    """Save scene flow vectors as PLY with flow as colors"""
    points = points.cpu().numpy()
    flow = flow.cpu().numpy()
    
    # Reshape if needed  
    if len(points.shape) == 4:  # Bx3xHxW
        B, C, H, W = points.shape
        points = points.reshape(B, C, -1).transpose(0, 2, 1).reshape(-1, C)
    elif len(points.shape) == 3:  # BxNx3
        B, N, C = points.shape
        points = points.reshape(-1, C)
    
    if len(flow.shape) == 4:  # Bx3xHxW
        B, C, H, W = flow.shape
        flow = flow.reshape(B, C, -1).transpose(0, 2, 1).reshape(-1, C)
    elif len(flow.shape) == 3:  # BxNx3
        B, N, C = flow.shape
        flow = flow.reshape(-1, C)
    
    # Remove invalid points
    valid_mask = points[:, 2] > 0
    points = points[valid_mask]
    flow = flow[valid_mask]
    
    # Normalize flow magnitudes for color mapping (0-255)
    flow_magnitude = np.linalg.norm(flow, axis=1)
    max_flow = np.percentile(flow_magnitude, 95)  # Use 95th percentile to avoid outliers
    flow_colors = (flow_magnitude / (max_flow + 1e-8) * 255).astype(np.uint8)
    flow_colors = np.stack([flow_colors, flow_colors, flow_colors], axis=1)
    
    print(f"Saving {len(points)} points with scene flow to {filename}")
    print(f"Flow magnitude range: {flow_magnitude.min():.6f} - {flow_magnitude.max():.6f}")
    
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n") 
        f.write("property float flow_x\n")
        f.write("property float flow_y\n")
        f.write("property float flow_z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            f.write(f"{points[i,0]:.6f} {points[i,1]:.6f} {points[i,2]:.6f} "
                   f"{flow[i,0]:.6f} {flow[i,1]:.6f} {flow[i,2]:.6f} "
                   f"{flow_colors[i,0]} {flow_colors[i,1]} {flow_colors[i,2]}\n")


def rescale_to_original_aspect_ratio(points_3d, scene_flow, img_colors, orig_size, model_size):
    """Rescale point cloud and scene flow to match original image aspect ratio"""
    orig_width, orig_height = orig_size
    model_width, model_height = model_size
    
    print(f"Rescaling from model size {model_width}x{model_height} to original aspect ratio {orig_width}x{orig_height}")
    
    # Calculate scaling factors
    scale_x = orig_width / model_width
    scale_y = orig_height / model_height
    
    print(f"Scaling factors: x={scale_x:.3f}, y={scale_y:.3f}")
    
    # Clone tensors to avoid modifying originals
    points_3d = points_3d.clone()
    scene_flow = scene_flow.clone()
    img_colors = img_colors.clone()
    
    # Rescale X and Y coordinates of point cloud (Z stays the same)
    points_3d[..., 0] *= scale_x  # X coordinates
    points_3d[..., 1] *= scale_y  # Y coordinates
    # points_3d[..., 2] unchanged (depth)
    
    # Rescale X and Y components of scene flow 
    scene_flow[:, 0] *= scale_x  # X flow component
    scene_flow[:, 1] *= scale_y  # Y flow component  
    # scene_flow[:, 2] unchanged (Z flow component)
    
    # For colors, we need to be more careful about maintaining spatial correspondence
    # We'll interpolate the color array to match the rescaled coordinate system
    B, C, H, W = img_colors.shape
    
    # Calculate target dimensions that maintain original aspect ratio
    target_aspect = orig_width / orig_height
    current_aspect = W / H
    
    if target_aspect > current_aspect:
        # Target is wider - expand width
        new_width = int(H * target_aspect)
        new_height = H
    else:
        # Target is taller - expand height  
        new_height = int(W / target_aspect)
        new_width = W
    
    # Interpolate colors to new dimensions
    img_colors_rescaled = torch.nn.functional.interpolate(
        img_colors, 
        size=(new_height, new_width), 
        mode='bilinear', 
        align_corners=True
    )
    
    print(f"Resized colors from {B}x{C}x{H}x{W} to {img_colors_rescaled.shape}")
    
    # Update points_3d shape to match color dimensions if needed
    B, H_orig, W_orig, _ = points_3d.shape
    if new_height != H_orig or new_width != W_orig:
        # Interpolate point cloud to match new color dimensions
        points_3d_reshaped = points_3d.permute(0, 3, 1, 2)  # BxHxWx3 -> Bx3xHxW
        points_3d_resized = torch.nn.functional.interpolate(
            points_3d_reshaped, 
            size=(new_height, new_width), 
            mode='bilinear', 
            align_corners=True
        )
        points_3d = points_3d_resized.permute(0, 2, 3, 1)  # Bx3xHxW -> BxHxWx3
        
        # Also resize scene flow to match
        scene_flow_resized = torch.nn.functional.interpolate(
            scene_flow,
            size=(new_height, new_width),
            mode='bilinear',
            align_corners=True
        )
        scene_flow = scene_flow_resized
        
        print(f"Resized point cloud to match color dimensions: {points_3d.shape}")
        print(f"Resized scene flow to match: {scene_flow.shape}")
    
    return points_3d, scene_flow, img_colors_rescaled


@torch.no_grad()
def run_inference(model, view1, view2, output_dir):
    """Run model inference and save results"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Get original image dimensions
    orig_size1 = view1.get('orig_size', (512, 384))
    model_size = (512, 384)  # Model input size
    
    # Prepare batch
    batch = list_to_batch([view1, view2], device)
    
    # Remove camera intrinsics to skip scaling (model will estimate focal length)
    batch[0].pop("camera_intrinsics", None)
    batch[1].pop("camera_intrinsics", None)
    
    # Prepare input dictionary
    input_dict = {
        "view1": batch[0],
        "view2": batch[1], 
        "camera_space": model.camera_space,
        "first_camera": model.first_camera,
        "align_scale": False  # Don't scale since we don't have GT
    }
    
    print("Running model inference...")
    output_dict = model.forward(input_dict, batch, None)
    
    # Get point cloud (left camera frame)
    points_3d = output_dict["xyz_1"]  # BxHxWx3
    print(f"Point cloud shape: {points_3d.shape}")
    
    # Get scene flow 
    scene_flow = model.get_sf(batch, output_dict)  # Bx3xHxW
    print(f"Scene flow shape: {scene_flow.shape}")
    
    # Get colors from first image
    img_colors = batch[0]["img"]  # Bx3xHxW
    
    # Rescale to original aspect ratio
    points_3d_rescaled, scene_flow_rescaled, img_colors_rescaled = rescale_to_original_aspect_ratio(
        points_3d, scene_flow, img_colors, orig_size1, model_size
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save rescaled point cloud with colors
    points_filename = os.path.join(output_dir, "point_cloud.ply")
    save_point_cloud(points_3d_rescaled.permute(0, 3, 1, 2), img_colors_rescaled, points_filename)
    
    # Save rescaled scene flow
    flow_filename = os.path.join(output_dir, "scene_flow.ply")
    save_scene_flow(points_3d_rescaled.permute(0, 3, 1, 2), scene_flow_rescaled, flow_filename)
    
    # Save rescaled numpy arrays for further processing
    np.save(os.path.join(output_dir, "points_3d.npy"), points_3d_rescaled.cpu().numpy())
    np.save(os.path.join(output_dir, "scene_flow.npy"), scene_flow_rescaled.cpu().numpy())
    np.save(os.path.join(output_dir, "img_colors.npy"), img_colors_rescaled.cpu().numpy())
    
    # Also save original (non-rescaled) data for comparison
    np.save(os.path.join(output_dir, "points_3d_original.npy"), points_3d.cpu().numpy())
    np.save(os.path.join(output_dir, "scene_flow_original.npy"), scene_flow.cpu().numpy())
    np.save(os.path.join(output_dir, "img_colors_original.npy"), img_colors.cpu().numpy())
    
    print(f"Results saved to: {output_dir}")
    print(f"  - Point cloud (rescaled): {points_filename}")
    print(f"  - Scene flow (rescaled): {flow_filename}")
    print(f"  - Rescaled arrays: points_3d.npy, scene_flow.npy, img_colors.npy")
    print(f"  - Original arrays: points_3d_original.npy, scene_flow_original.npy, img_colors_original.npy")


def main():
    parser = argparse.ArgumentParser(description='Demo script for stereo scene flow inference')
    parser.add_argument('--model', required=True, help='Path to model config YAML')
    parser.add_argument('--data', required=True, help='Path to folder containing two images')
    parser.add_argument('--output', default='demo_output', help='Output directory')
    parser.add_argument('--focal', type=float, help='Camera focal length (optional)')
    
    args = parser.parse_args()
    
    # Load model config
    with open(args.model, 'r') as f:
        model_config = yaml.safe_load(f)
    
    print(f"Loading model from config: {args.model}")
    model = instantiate_component(model_config)
    
    # Find two images in data folder
    data_path = Path(args.data)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in data_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if len(image_files) < 2:
        raise ValueError(f"Need at least 2 images in {data_path}, found {len(image_files)}")
    
    # Sort and take first two
    image_files.sort()
    image1_path = image_files[0] 
    image2_path = image_files[1]
    
    print(f"Using images:")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")
    
    # Prepare input data (model expects 512x384 input size)
    view1, view2 = prepare_input_data(image1_path, image2_path, target_size=(512, 384))
    
    # Override focal length if provided
    if args.focal is not None:
        intrinsics = view1['camera_intrinsics'].numpy()
        intrinsics[0, 0] = args.focal
        intrinsics[1, 1] = args.focal
        view1['camera_intrinsics'] = torch.from_numpy(intrinsics)
        view2['camera_intrinsics'] = torch.from_numpy(intrinsics)
        print(f"Using focal length: {args.focal}")
    
    # Run inference
    run_inference(model, view1, view2, args.output)

if __name__ == "__main__":
    main()
