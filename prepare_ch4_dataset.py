#!/usr/bin/env python3
"""
Prepare 4-channel datasets by combining RGB images with motion difference masks.

This script combines 3-channel RGB images from datasets/phantom00-39 and datasets/phantom40-140
with their corresponding motion difference masks to create 4-channel TIFF images.
The output follows YOLO dataset format with new datasets: ch4_phantom00-39 and ch4_phantom40-140.

Channel layout: [B, G, R, MotionDiff] (OpenCV default BGR format + Motion Difference)
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


def combine_rgb_movediff(rgb_path, mask_path):
    """Combine RGB image with motion difference mask into 4-channel image."""
    # Load RGB image (BGR format by default)
    bgr_img = cv2.imread(str(rgb_path))
    if bgr_img is None:
        raise ValueError(f"Could not load RGB image: {rgb_path}")
    
    # Load motion difference mask (grayscale)
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        raise ValueError(f"Could not load mask image: {mask_path}")
    
    # Ensure both images have the same dimensions
    if bgr_img.shape[:2] != mask_img.shape:
        # Resize mask to match RGB image dimensions
        mask_img = cv2.resize(mask_img, (bgr_img.shape[1], bgr_img.shape[0]))
    
    # Combine into 4-channel image [B, G, R, MotionDiff] (keeping BGR format)
    h, w = bgr_img.shape[:2]
    ch4_img = np.zeros((h, w, 4), dtype=np.uint8)
    ch4_img[:, :, :3] = bgr_img  # BGR channels
    ch4_img[:, :, 3] = mask_img  # Motion difference channel
    
    return ch4_img


def save_tiff(image, output_path):
    """Save 4-channel image as TIFF format using cv2.imwritemulti."""
    # Split the 4-channel image into separate channels
    b_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    r_channel = image[:, :, 2]
    motion_channel = image[:, :, 3]
    
    # Create list of images for multi-page TIFF
    pages = [b_channel, g_channel, r_channel, motion_channel]
    
    # Save as multi-page TIFF
    success = cv2.imwritemulti(str(output_path), pages)
    if not success:
        raise ValueError(f"Failed to save TIFF image: {output_path}")


def copy_labels(source_labels_dir, target_labels_dir):
    """Copy label files from source to target directory."""
    if not source_labels_dir.exists():
        print(f"Warning: Labels directory not found: {source_labels_dir}")
        return
    
    target_labels_dir.mkdir(parents=True, exist_ok=True)
    
    for label_file in source_labels_dir.glob("*.txt"):
        shutil.copy2(label_file, target_labels_dir / label_file.name)


def process_dataset(source_dataset_path, target_dataset_name):
    """Process a single dataset to create 4-channel version."""
    source_images_dir = source_dataset_path / "images"
    source_masks_dir = source_dataset_path / "masks" 
    source_labels_dir = source_dataset_path / "labels"
    
    # Create target dataset directory structure
    target_dataset_path = Path(__file__).parent / "datasets" / target_dataset_name
    target_images_dir = target_dataset_path / "images"
    target_labels_dir = target_dataset_path / "labels"
    
    target_images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing dataset: {source_dataset_path.name} -> {target_dataset_name}")
    
    # Check if source directories exist
    if not source_images_dir.exists():
        print(f"Error: Images directory not found: {source_images_dir}")
        return
    
    if not source_masks_dir.exists():
        print(f"Error: Masks directory not found: {source_masks_dir}")
        return
    
    # Get all image files
    image_files = list(source_images_dir.glob("*.jpg"))
    print(f"Found {len(image_files)} image files")
    
    # Process each image that has a corresponding mask
    processed_count = 0
    skipped_count = 0
    
    for rgb_path in tqdm(image_files, desc="Processing images"):
        # Find corresponding mask file
        mask_path = source_masks_dir / rgb_path.name
        
        if not mask_path.exists():
            skipped_count += 1
            continue  # Skip if no corresponding mask
        
        try:
            # Combine RGB and motion difference
            ch4_image = combine_rgb_movediff(rgb_path, mask_path)
            
            # Create output filename (replace .jpg with .tiff)
            output_filename = rgb_path.stem + ".tiff"
            output_path = target_images_dir / output_filename
            
            # Save as TIFF
            save_tiff(ch4_image, output_path)
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {rgb_path.name}: {e}")
            skipped_count += 1
    
    # Copy label files
    copy_labels(source_labels_dir, target_labels_dir)
    
    print(f"Processed: {processed_count} images")
    print(f"Skipped: {skipped_count} images")
    print(f"4-channel dataset created: {target_dataset_path}")


def create_dataset_yaml(dataset_name):
    """Create YAML configuration file for the dataset."""
    # Determine train/val paths based on dataset name
    if "phantom00-39" in dataset_name:
        train_path = f"datasets/{dataset_name.replace('ch4_', 'ch4_phantom40-140')}/images"
        val_path = f"datasets/{dataset_name}/images"
        test_path = f"datasets/{dataset_name}/images"
    elif "phantom40-140" in dataset_name:
        train_path = f"datasets/{dataset_name}/images"
        val_path = f"datasets/{dataset_name.replace('ch4_', 'ch4_phantom00-39')}/images"
        test_path = f"datasets/{dataset_name.replace('ch4_', 'ch4_phantom00-39')}/images"
    else:
        # Default fallback
        train_path = f"datasets/{dataset_name}/images"
        val_path = f"datasets/{dataset_name}/images"
        test_path = f"datasets/{dataset_name}/images"
    
    yaml_content = f"""train: {train_path}
val: {val_path}
test: {test_path}

channels: 4

names:
  0: ball
"""
    
    yaml_path = Path(__file__).parent / "data" / f"{dataset_name}.yaml"
    yaml_path.parent.mkdir(exist_ok=True)
    
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset configuration saved: {yaml_path}")


def main():
    """Main function to process both datasets."""
    # Define source datasets and their target names
    datasets_to_process = [
        ("phantom00-39", "ch4_phantom00-39"),
        ("phantom40-140", "ch4_phantom40-140")
    ]
    
    base_dir = Path(__file__).parent / "datasets"
    
    print("Starting 4-channel dataset preparation...")
    print("=" * 50)
    
    for source_name, target_name in datasets_to_process:
        source_path = base_dir / source_name
        
        if source_path.exists():
            process_dataset(source_path, target_name)
            create_dataset_yaml(target_name)
            print()
        else:
            print(f"Source dataset not found: {source_path}")
    
    print("4-channel dataset preparation completed!")
    print("\nUsage:")
    print("  python train.py --data data/ch4_phantom00-39.yaml --weights yolov5s.pt --img 640")
    print("  python detect_ch.py --weights runs/train/exp/weights/best.pt --source datasets/ch4_phantom40-140/images/")


if __name__ == "__main__":
    main()