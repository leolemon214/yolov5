#!/usr/bin/env python3
"""
Generate motion difference images from phantom dataset sequences.

This script processes images in datasets/phantom00-39 and datasets/phantom40-140,
calculates motion difference between consecutive frames using the method from utils/movediff.py,
and saves the results as grayscale JPG images in masks/ directories.

The last 4 digits in filenames represent frame numbers. Motion difference is calculated
between consecutive frames, so the first frame of each sequence is skipped.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import re
from tqdm import tqdm

# Import motion compensation from utils
import sys
sys.path.append(str(Path(__file__).parent))
from utils.movediff import motion_compensate


def generate_motion_difference_2frame(frame1, frame2):
    """
    Calculate motion difference between two consecutive frames.
    Simplified version using only 2 frames instead of 3.
    """
    # Apply Gaussian blur and convert to grayscale
    frame1_blur = cv2.GaussianBlur(frame1, (11, 11), 0)
    frame1_gray = cv2.cvtColor(frame1_blur, cv2.COLOR_BGR2GRAY)
    
    frame2_blur = cv2.GaussianBlur(frame2, (11, 11), 0)
    frame2_gray = cv2.cvtColor(frame2_blur, cv2.COLOR_BGR2GRAY)
    
    # Motion compensation
    compensated_frame1, _, _, _, _, _ = motion_compensate(frame1_gray, frame2_gray)
    
    # Calculate absolute difference
    frame_diff = cv2.absdiff(frame2_gray, compensated_frame1)
    
    return frame_diff


def extract_frame_number(filename):
    """Extract frame number from filename (last 4 digits before extension)."""
    match = re.search(r'(\d{4})\.jpg$', filename)
    if match:
        return int(match.group(1))
    return None


def get_consecutive_frames(image_dir):
    """Get list of consecutive frame pairs for motion difference calculation."""
    # Get all image files and sort them
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # Group by sequence prefix and sort by frame number
    sequences = {}
    for filename in image_files:
        frame_num = extract_frame_number(filename)
        if frame_num is not None:
            prefix = filename[:-8]  # Remove last 4 digits and .jpg
            if prefix not in sequences:
                sequences[prefix] = []
            sequences[prefix].append((frame_num, filename))
    
    # Sort each sequence by frame number
    for prefix in sequences:
        sequences[prefix].sort(key=lambda x: x[0])
    
    # Generate pairs for each sequence
    pairs = []
    for prefix, frames in sequences.items():
        if len(frames) >= 2:
            for i in range(1, len(frames)):
                frame1_num, frame1_file = frames[i-1]
                frame2_num, frame2_file = frames[i]
                
                # Check if frames are consecutive
                if frame2_num == frame1_num + 1:
                    pairs.append((frame1_file, frame2_file))
    
    return pairs


def process_dataset(dataset_path, brightness_gain=3.0):
    """Process a single dataset to generate motion difference images."""
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"
    
    if not images_dir.exists():
        print(f"Images directory not found: {images_dir}")
        return
    
    # Create masks directory if it doesn't exist
    masks_dir.mkdir(exist_ok=True)
    
    print(f"Processing dataset: {dataset_path.name}")
    
    # Get consecutive frame pairs
    pairs = get_consecutive_frames(images_dir)
    print(f"Found {len(pairs)} frame pairs for motion difference calculation")
    
    if len(pairs) == 0:
        print("No consecutive frame pairs found")
        return
    
    # Process each pair
    for frame1_file, frame2_file in tqdm(pairs, desc="Generating motion differences"):
        try:
            # Load the two consecutive frames
            frame1_path = images_dir / frame1_file
            frame2_path = images_dir / frame2_file
            
            frame1 = cv2.imread(str(frame1_path))
            frame2 = cv2.imread(str(frame2_path))
            
            if frame1 is None or frame2 is None:
                print(f"Failed to load frames: {frame1_file}, {frame2_file}")
                continue
            
            # Calculate motion difference using 2-frame method
            motion_diff = generate_motion_difference_2frame(frame1, frame2)
            
            # Apply brightness enhancement
            motion_diff = np.clip(motion_diff.astype(np.float32) * brightness_gain, 0, 255).astype(np.uint8)
            
            # Save motion difference image (named after the second frame)
            output_filename = frame2_file  # Same name as second frame
            output_path = masks_dir / output_filename
            
            cv2.imwrite(str(output_path), motion_diff)
            
        except Exception as e:
            print(f"Error processing pair {frame1_file}, {frame2_file}: {e}")
            continue
    
    print(f"Motion difference images saved to: {masks_dir}")


def main():
    """Main function to process both datasets."""
    # Define dataset paths
    base_dir = Path(__file__).parent / "datasets"
    
    datasets = [
        base_dir / "phantom00-39",
        base_dir / "phantom40-140"
    ]
    
    print("Starting motion difference generation...")
    print("=" * 50)
    
    for dataset_path in datasets:
        if dataset_path.exists():
            process_dataset(dataset_path, brightness_gain=3.0)
            print()
        else:
            print(f"Dataset not found: {dataset_path}")
    
    print("Motion difference generation completed!")


if __name__ == "__main__":
    main()