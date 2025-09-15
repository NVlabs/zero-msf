# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-SA-4.0
#
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License. https://creativecommons.org/licenses/by-nc-sa/4.0/


import os
import time
import argparse
from pathlib import Path
import threading

import numpy as np
import viser


class SimpleSceneFlowVisualizer:
    def __init__(self, server: viser.ViserServer, num_steps: int = 50):
        self.server = server
        self.num_steps = num_steps
        
        # Data
        self.points_3d = None
        self.scene_flow = None 
        self.img_colors = None
        self.point_cloud_handle = None
        
        # Animation state
        self.current_step = 0
        self.is_playing = False
        
        # Setup UI
        self.setup_ui()
        
    def setup_ui(self):
        """Setup simple UI controls"""
        with self.server.gui.add_folder("Scene Flow Animation"):
            # Play/Pause controls
            self.play_button = self.server.gui.add_button("Play")
            self.pause_button = self.server.gui.add_button("Pause") 
            self.reset_button = self.server.gui.add_button("Reset")
            
            # Step slider
            self.step_slider = self.server.gui.add_slider(
                "Animation Step",
                min=0,
                max=self.num_steps,
                step=1,
                initial_value=0
            )
            
            # Point size control
            self.point_size_slider = self.server.gui.add_slider(
                "Point Size",
                min=0.001,
                max=0.1,
                step=0.001,
                initial_value=0.008
            )
        
        # Callbacks
        self.play_button.on_click(self._on_play)
        self.pause_button.on_click(self._on_pause)
        self.reset_button.on_click(self._on_reset)
        self.step_slider.on_update(self._on_step_update)
        self.point_size_slider.on_update(self._on_point_size_update)
    
    def load_data(self, data_dir: str):
        """Load point cloud and scene flow data"""
        data_path = Path(data_dir)
        
        # Load numpy arrays
        self.points_3d = np.load(data_path / "points_3d.npy")  # BxHxWx3
        self.scene_flow = np.load(data_path / "scene_flow.npy")  # Bx3xHxW
        self.img_colors = np.load(data_path / "img_colors.npy")  # Bx3xHxW
        
        print(f"Loaded data shapes:")
        print(f"  Points: {self.points_3d.shape}")
        print(f"  Flow: {self.scene_flow.shape}")
        print(f"  Colors: {self.img_colors.shape}")
        
        # Reshape and process data
        B, H, W, _ = self.points_3d.shape
        
        # Flatten spatial dimensions: BxHxWx3 -> BxNx3
        self.points_3d = self.points_3d.reshape(B, -1, 3)
        
        # Convert flow from Bx3xHxW to BxNx3
        self.scene_flow = self.scene_flow.transpose(0, 2, 3, 1).reshape(B, -1, 3)
        
        # Convert colors from Bx3xHxW to BxNx3
        self.img_colors = self.img_colors.transpose(0, 2, 3, 1).reshape(B, -1, 3)
        
        # Handle batch dimension (assume batch size 1)
        if B == 1:
            self.points_3d = self.points_3d[0]  # Nx3
            self.scene_flow = self.scene_flow[0]  # Nx3
            self.img_colors = self.img_colors[0]  # Nx3
        
        # Filter valid points (depth > 0)
        valid_mask = self.points_3d[:, 2] > 0
        self.points_3d = self.points_3d[valid_mask]
        self.scene_flow = self.scene_flow[valid_mask]
        self.img_colors = self.img_colors[valid_mask]
        
        print(f"Using {len(self.points_3d)} valid points")
        
        # Normalize colors to [0,1] range
        if self.img_colors.max() > 1.0:
            self.img_colors = self.img_colors / 255.0
        
        # Update slider max value
        self.step_slider.max = self.num_steps
        
        # Show initial frame
        self.update_visualization(0)
    
    def get_interpolated_points(self, step: int):
        """Get interpolated point positions for given step"""
        if self.points_3d is None:
            return None
            
        # Convert step to interpolation factor t ∈ [0, 1]
        t = step / self.num_steps
        
        # Linear interpolation: p(t) = p0 + t * flow
        interpolated_points = self.points_3d + t * self.scene_flow
        
        return interpolated_points
    
    def update_visualization(self, step: int):
        """Update point cloud visualization for given step"""
        if self.points_3d is None:
            return
        
        # Get interpolated points
        current_points = self.get_interpolated_points(step)
        
        # Remove existing point cloud
        if self.point_cloud_handle is not None:
            self.point_cloud_handle.remove()
        
        # Add new point cloud
        self.point_cloud_handle = self.server.scene.add_point_cloud(
            "/point_cloud",
            points=current_points,
            colors=self.img_colors,
            point_size=self.point_size_slider.value
        )
    
    def _on_play(self, _):
        """Start animation"""
        self.is_playing = True
        if not hasattr(self, '_animation_thread') or not self._animation_thread.is_alive():
            self._animation_thread = threading.Thread(target=self._animation_loop)
            self._animation_thread.daemon = True
            self._animation_thread.start()
    
    def _on_pause(self, _):
        """Pause animation"""
        self.is_playing = False
    
    def _on_reset(self, _):
        """Reset to first frame"""
        self.is_playing = False
        self.current_step = 0
        self.step_slider.value = 0
        self.update_visualization(0)
    
    def _on_step_update(self, _):
        """Handle manual step slider update"""
        self.current_step = self.step_slider.value
        self.update_visualization(self.current_step)
    
    def _on_point_size_update(self, _):
        """Handle point size update"""
        if self.point_cloud_handle is not None:
            self.point_cloud_handle.point_size = self.point_size_slider.value
    
    def _animation_loop(self):
        """Animation loop"""
        while True:
            if self.is_playing and self.points_3d is not None:
                # Advance to next step
                self.current_step = (self.current_step + 1) % (self.num_steps + 1)
                
                # Update UI and visualization
                self.step_slider.value = self.current_step
                self.update_visualization(self.current_step)
                
                # Sleep for animation speed (adjust as needed)
                time.sleep(0.1)  # 10 FPS
            else:
                time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description='Simple scene flow visualization')
    parser.add_argument('--data', required=True, help='Path to demo output directory')
    parser.add_argument('--steps', type=int, default=50, help='Number of interpolation steps')
    parser.add_argument('--port', type=int, default=8080, help='Viser server port')
    
    args = parser.parse_args()
    
    # Check data directory
    if not os.path.exists(args.data):
        print(f"Error: Data directory {args.data} does not exist")
        return
    
    # Check required files
    required_files = ['points_3d.npy', 'scene_flow.npy', 'img_colors.npy']
    for file in required_files:
        if not os.path.exists(os.path.join(args.data, file)):
            print(f"Error: Required file {file} not found in {args.data}")
            return
    
    try:
        print(f"Starting visualization server on port {args.port}...")
        server = viser.ViserServer(port=args.port)
        
        print("Creating visualizer...")
        visualizer = SimpleSceneFlowVisualizer(server, args.steps)
        
        print("Loading data...")
        visualizer.load_data(args.data)
        
        print(f"\n🎉 Visualization ready!")
        print(f"📍 Open http://localhost:{args.port} in your browser")
        print("🎮 Controls:")
        print("   - Play: Start animation")
        print("   - Pause: Stop animation") 
        print("   - Reset: Go to first frame")
        print("   - Animation Step: Manually scrub through frames")
        print("   - Point Size: Adjust point visibility")
        print(f"\n📊 Animation: {args.steps} steps from initial to final positions")
        print("\nPress Ctrl+C to stop...")
        
        # Keep server running
        while True:
            time.sleep(0.1)
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    except KeyboardInterrupt:
        print("\nShutting down...")

if __name__ == "__main__":
    main()
