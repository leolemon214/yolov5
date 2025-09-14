#!/usr/bin/env python3
"""
Synthetic golf ball data generator.

Generates synthetic training data by simulating golf balls dropping from the sky
with realistic trajectories and timing.
"""

import os
import cv2
import numpy as np
import random
from pathlib import Path
import shutil
from tqdm import tqdm
import argparse


class SyntheticDataGenerator:
    """Main class for generating synthetic golf ball data."""
    
    def __init__(self, source_dataset_path, target_dataset_path, 
                 diameter_range=(4, 15), max_balls=2, bbox_expand_ratio=1.5):
        self.source_path = Path(source_dataset_path)
        self.target_path = Path(target_dataset_path)
        self.diameter_range = diameter_range
        self.max_balls = max_balls
        self.bbox_expand_ratio = bbox_expand_ratio
        
        # Create target directory structure
        (self.target_path / "images").mkdir(parents=True, exist_ok=True)
        (self.target_path / "labels").mkdir(parents=True, exist_ok=True)
        
    def _load_original_labels(self, label_path):
        """Load original YOLO format labels."""
        labels = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        class_id = int(parts[0])
                        bbox = [float(x) for x in parts[1:5]]
                        labels.append((class_id, bbox))
        return labels
        
    def _simulate_sequence(self, image_files, start_frame):
        """Simulate balls dropping one by one with random intervals."""
        if start_frame >= len(image_files):
            return []
            
        # Load first image to get dimensions
        first_img_path = self.source_path / "images" / image_files[start_frame]
        first_img = cv2.imread(str(first_img_path))
        if first_img is None:
            return []
            
        img_height, img_width = first_img.shape[:2]
        max_frames = min(120, len(image_files) - start_frame)
        ball_trajectories = []
        
        # Generate ball drop schedule with max_balls constraint
        next_ball_frame = 0
        
        while next_ball_frame < max_frames - 45:
            # Check if adding a new ball would exceed max_balls limit
            active_count = sum(1 for traj in ball_trajectories 
                             if next_ball_frame >= traj['start_frame'] and 
                                next_ball_frame < traj['start_frame'] + traj['duration'])
            
            if active_count >= self.max_balls:
                next_ball_frame += 5
                continue
            
            # Create new ball trajectory
            drop_duration = random.randint(10, 45)  # 0.33-1.5 seconds at 30fps
            start_x = random.randint(50, img_width - 50)
            start_y = -random.randint(50, 150)
            end_x = random.randint(50, img_width - 50)
            end_y = random.randint(int(img_height * 0.7), int(img_height * 0.9))
            
            # Calculate velocity for motion blur
            velocity_x = (end_x - start_x) / drop_duration
            velocity_y = (end_y - start_y) / drop_duration
            
            trajectory = {
                'start_frame': next_ball_frame,
                'duration': drop_duration,
                'diameter': random.randint(*self.diameter_range),
                'start_x': start_x,
                'start_y': start_y,
                'end_x': end_x,
                'end_y': end_y,
                'velocity_x': velocity_x,
                'velocity_y': velocity_y,
                'brightness': random.uniform(0.8, 1.2),
                'blur': random.randint(0, 2),
            }
            ball_trajectories.append(trajectory)
            
            next_ball_frame += random.randint(10, 30)  # Random interval
        
        # Generate frame data
        frame_data = []
        for frame_offset in range(max_frames):
            frame_balls = []
            
            for traj in ball_trajectories:
                if (frame_offset >= traj['start_frame'] and 
                    frame_offset < traj['start_frame'] + traj['duration']):
                    
                    # Calculate current position
                    progress = (frame_offset - traj['start_frame']) / traj['duration']
                    current_x = traj['start_x'] + (traj['end_x'] - traj['start_x']) * progress
                    current_y = traj['start_y'] + (traj['end_y'] - traj['start_y']) * progress
                    
                    # Check if ball is visible
                    radius = traj['diameter'] // 2
                    if (current_y > -radius and current_y < img_height + radius and
                        current_x > -radius and current_x < img_width + radius):
                        
                        frame_balls.append({
                            'x': current_x,
                            'y': current_y,
                            'radius': radius,
                            'brightness': traj['brightness'],
                            'blur': traj['blur'],
                            'velocity_x': traj['velocity_x'],
                            'velocity_y': traj['velocity_y'],
                        })
            
            frame_data.append({
                'frame_idx': start_frame + frame_offset,
                'balls': frame_balls
            })
        
        return frame_data
    
    def _draw_ball(self, img, x, y, radius, brightness, blur, velocity_x=0, velocity_y=0):
        """Draw a single ball on the image with motion blur effect."""
        img_height, img_width = img.shape[:2]
        center = (int(x), int(y))
        draw_radius = max(1, radius)
        
        # Skip if outside bounds
        if (center[0] < -draw_radius or center[0] > img_width + draw_radius or
            center[1] < -draw_radius or center[1] > img_height + draw_radius):
            return False
        
        # Calculate motion blur parameters
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        motion_blur_length = min(int(speed * 0.3), draw_radius * 2)  # Scale blur with speed
        
        # Draw main ball with motion blur effect
        base_color = int(255 * brightness)
        color = (base_color, base_color, base_color)
        
        if motion_blur_length > 1:
            # Create motion blur by drawing multiple circles along motion path
            num_trails = max(3, min(motion_blur_length // 2, 8))  # 3-8 trail circles
            
            # Calculate direction vector
            if speed > 0:
                direction_x = velocity_x / speed
                direction_y = velocity_y / speed
            else:
                direction_x = direction_y = 0
            
            # Draw trail circles with decreasing opacity using alpha blending
            for i in range(num_trails):
                trail_factor = (num_trails - i - 1) / num_trails  # 1.0 to 0.0
                trail_offset = motion_blur_length * trail_factor
                
                trail_x = int(x - direction_x * trail_offset)
                trail_y = int(y - direction_y * trail_offset)
                trail_center = (trail_x, trail_y)
                
                # Check bounds for trail circle
                if (trail_center[0] >= -draw_radius and trail_center[0] <= img_width + draw_radius and
                    trail_center[1] >= -draw_radius and trail_center[1] <= img_height + draw_radius):
                    
                    # Calculate opacity for this trail circle
                    trail_opacity = 0.1 + 0.7 * (i / num_trails)  # 0.1 to 0.8
                    trail_radius = max(1, int(draw_radius * (0.8 + 0.2 * trail_opacity)))
                    
                    # Create overlay for alpha blending
                    overlay = img.copy()
                    cv2.circle(overlay, trail_center, trail_radius, color, -1)
                    
                    # Blend overlay with original image using alpha
                    cv2.addWeighted(overlay, trail_opacity, img, 1 - trail_opacity, 0, img)
        else:
            # No motion blur, draw normal circle
            cv2.circle(img, center, draw_radius, color, -1)
        
        # Draw main circle (always on top)
        cv2.circle(img, center, draw_radius, color, -1)
        
        # Add subtle border for visibility
        if draw_radius > 2:
            cv2.circle(img, center, draw_radius + 1, (128, 128, 128), 1)
        
        # Apply additional gaussian blur if specified
        if blur > 0:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.circle(mask, center, draw_radius + motion_blur_length + 2, 255, -1)
            kernel_size = max(3, blur * 2 + 1)
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            img[mask > 0] = blurred[mask > 0]
        
        return True
    
    def _generate_labels(self, balls, img_width, img_height):
        """Generate YOLO format labels for synthetic balls."""
        labels = []
        for ball in balls:
            x, y, radius = ball['x'], ball['y'], ball['radius']
            
            # Calculate expanded bounding box
            expanded_radius = radius * self.bbox_expand_ratio
            x1 = max(0, int(x - expanded_radius))
            y1 = max(0, int(y - expanded_radius))
            x2 = min(img_width, int(x + expanded_radius))
            y2 = min(img_height, int(y + expanded_radius))
            
            # Convert to YOLO format
            center_x = (x1 + x2) / 2 / img_width
            center_y = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            labels.append((0, [center_x, center_y, width, height]))
        
        return labels
        
    def generate(self):
        """Generate synthetic data for the entire dataset."""
        source_images_dir = self.source_path / "images"
        source_labels_dir = self.source_path / "labels"
        
        if not source_images_dir.exists():
            print(f"Source images directory not found: {source_images_dir}")
            return
            
        image_files = sorted([f.name for f in source_images_dir.glob("*.jpg")])
        print(f"Found {len(image_files)} images to process")
        
        processed_frames = set()
        sequence_start = 0
        
        with tqdm(total=len(image_files), desc="Generating synthetic data") as pbar:
            while sequence_start < len(image_files):
                frame_data = self._simulate_sequence(image_files, sequence_start)
                
                if not frame_data:
                    sequence_start += 1
                    continue
                    
                # Process each frame in the sequence
                for data in frame_data:
                    frame_idx = data['frame_idx']
                    balls = data['balls']
                    
                    if frame_idx >= len(image_files) or frame_idx in processed_frames:
                        continue
                        
                    # Load original image
                    img_file = image_files[frame_idx]
                    img_path = source_images_dir / img_file
                    label_file = img_file.replace('.jpg', '.txt')
                    label_path = source_labels_dir / label_file
                    
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue
                        
                    img_height, img_width = img.shape[:2]
                    
                    # Load original labels
                    original_labels = self._load_original_labels(label_path)
                    
                    # Draw synthetic balls with motion blur
                    for ball in balls:
                        self._draw_ball(img, ball['x'], ball['y'], ball['radius'], 
                                      ball['brightness'], ball['blur'],
                                      ball['velocity_x'], ball['velocity_y'])
                    
                    # Save modified image
                    target_img_path = self.target_path / "images" / img_file
                    cv2.imwrite(str(target_img_path), img)
                    
                    # Generate and save labels
                    synthetic_labels = self._generate_labels(balls, img_width, img_height)
                    target_label_path = self.target_path / "labels" / label_file
                    
                    with open(target_label_path, 'w') as f:
                        # Write original labels
                        for class_id, bbox in original_labels:
                            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                        # Write synthetic ball labels
                        for class_id, bbox in synthetic_labels:
                            f.write(f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
                    
                    processed_frames.add(frame_idx)
                    pbar.update(1)
                    
                sequence_start += max(1, len(frame_data) - 50)
                
            # Copy remaining frames without synthetic balls
            for i, img_file in enumerate(image_files):
                if i not in processed_frames:
                    # Copy original image and labels
                    shutil.copy2(source_images_dir / img_file, 
                               self.target_path / "images" / img_file)
                    
                    label_file = img_file.replace('.jpg', '.txt')
                    src_label = source_labels_dir / label_file
                    dst_label = self.target_path / "labels" / label_file
                    
                    if src_label.exists():
                        shutil.copy2(src_label, dst_label)
                    else:
                        dst_label.touch()
                        
                    pbar.update(1)
                    
        print(f"Synthetic dataset created: {self.target_path}")
        print(f"Processed {len(processed_frames)} frames with synthetic balls")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Generate synthetic golf ball training data')
    parser.add_argument('--source', default='datasets/phantom40-140',
                        help='Source dataset path')
    parser.add_argument('--target', default='datasets/synth_phantom40-140',
                        help='Target dataset path')
    parser.add_argument('--diameter-min', type=int, default=3,
                        help='Minimum ball diameter in pixels')
    parser.add_argument('--diameter-max', type=int, default=12,
                        help='Maximum ball diameter in pixels')
    parser.add_argument('--max-balls', type=int, default=4,
                        help='Maximum number of simultaneous balls')
    parser.add_argument('--bbox-expand', type=float, default=1.5,
                        help='Bbox expansion ratio relative to ball diameter')
    
    args = parser.parse_args()
    
    generator = SyntheticDataGenerator(
        source_dataset_path=args.source,
        target_dataset_path=args.target,
        diameter_range=(args.diameter_min, args.diameter_max),
        max_balls=args.max_balls,
        bbox_expand_ratio=args.bbox_expand
    )
    
    print("Starting synthetic data generation...")
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print(f"Ball diameter range: {args.diameter_min}-{args.diameter_max} pixels")
    print(f"Max simultaneous balls: {args.max_balls}")
    print(f"Bbox expansion ratio: {args.bbox_expand}")
    print("=" * 50)
    
    generator.generate()
    print("Synthetic data generation completed!")


if __name__ == "__main__":
    main()