#!/usr/bin/env python3
"""
Script to remove SIGF images from label 1 onwards for longitudinal data analysis.

This script processes SIGF dataset by:
1. Reading label files to find the first occurrence of label "1" (progression)
2. Removing corresponding images from that time point onwards
3. Keeping all labels intact (labels are not modified)

Note: No backup is created as the entire dataset is pre-backed up.

Usage: python remove_progression_images.py --data_dir /path/to/SIGF --split train/test/validation
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import glob


def read_label_file(label_path: str) -> List[int]:
    """Read label file and return list of labels."""
    with open(label_path, 'r') as f:
        labels = [int(line.strip()) for line in f if line.strip()]
    return labels


def find_first_progression_index(labels: List[int]) -> int:
    """Find the index of first progression label (1). Returns -1 if no progression."""
    try:
        return labels.index(1)
    except ValueError:
        return -1


def get_patient_images(image_dir: str, patient_eye_id: str) -> List[str]:
    """Get all images for a specific patient-eye combination, sorted by date."""
    # Check if images are in subdirectory (train structure) or direct directory (test/validation structure)
    subdir_path = os.path.join(image_dir, patient_eye_id)
    
    if os.path.exists(subdir_path):
        # Images are in patient-specific subdirectory
        # Pattern matches: SD1284_1990_04_11_OS.JPG (filename format in subdirectory)
        pattern = os.path.join(subdir_path, "*.jpg")
        pattern_upper = os.path.join(subdir_path, "*.JPG")
    else:
        # Images are directly in the image directory  
        # Pattern matches: SD1284_OS_*.jpg (expected format for direct files)
        pattern = os.path.join(image_dir, f"{patient_eye_id}_*.jpg") 
        pattern_upper = os.path.join(image_dir, f"{patient_eye_id}_*.JPG")
    
    images = glob.glob(pattern) + glob.glob(pattern_upper)
    images.sort()  # Sort by filename (which includes date)
    return images


def process_dataset_split(data_dir: str, split: str, dry_run: bool = True):
    """Process a single dataset split (train/test/validation)."""
    split_dir = os.path.join(data_dir, split)
    
    # Handle different directory structures
    if split == "train":
        image_dir = os.path.join(split_dir, "image", "all", "image")
        label_dir = os.path.join(split_dir, "label", "label")
    else:
        image_dir = os.path.join(split_dir, "image")
        label_dir = os.path.join(split_dir, "label")
    
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        print(f"Directory not found for split: {split}")
        return
    
    print(f"\nProcessing {split} split...")
    print(f"Image directory: {image_dir}")
    print(f"Label directory: {label_dir}")
    
    # Get all label files
    label_files = glob.glob(os.path.join(label_dir, "*.txt"))
    
    total_removed_images = 0
    processed_patients = 0
    
    for label_file in label_files:
        patient_eye_id = os.path.splitext(os.path.basename(label_file))[0]
        
        # Read labels
        labels = read_label_file(label_file)
        progression_index = find_first_progression_index(labels)
        
        if progression_index == -1:
            print(f"{patient_eye_id}: No progression detected, keeping all {len(labels)} images")
            continue
        
        # Get corresponding images
        images = get_patient_images(image_dir, patient_eye_id)
        
        if len(images) != len(labels):
            print(f"WARNING: {patient_eye_id} has {len(images)} images but {len(labels)} labels")
            continue
        
        # Identify images to remove (from progression_index onwards)
        images_to_remove = images[progression_index:]
        images_to_keep = images[:progression_index]
        
        print(f"{patient_eye_id}: Progression at index {progression_index}, removing {len(images_to_remove)} images, keeping {len(images_to_keep)}")
        print(f"  Labels remain unchanged ({len(labels)} labels kept)")
        
        if not dry_run:
            # Remove images directly (no backup needed)
            for img_path in images_to_remove:
                os.remove(img_path)
                print(f"  Removed: {os.path.basename(img_path)}")
        
        total_removed_images += len(images_to_remove)
        processed_patients += 1
    
    print(f"\nSummary for {split} split:")
    print(f"Processed patients: {processed_patients}")
    print(f"Total images removed: {total_removed_images}")
    if dry_run:
        print("(DRY RUN - no files were actually removed)")


def main():
    parser = argparse.ArgumentParser(description="Remove SIGF images from progression point onwards")
    parser.add_argument("--data_dir", required=True, help="Path to SIGF dataset directory")
    parser.add_argument("--split", choices=["train", "test", "validation", "all"], default="all",
                        help="Dataset split to process")
    parser.add_argument("--dry_run", action="store_true", default=True,
                        help="Perform dry run without actually removing files (default: True)")
    parser.add_argument("--execute", action="store_true",
                        help="Actually execute the removal (overrides dry_run)")
    
    args = parser.parse_args()
    
    # Override dry_run if execute is specified
    if args.execute:
        args.dry_run = False
    
    if args.dry_run:
        print("=== DRY RUN MODE ===")
        print("Use --execute flag to actually remove files")
        print("Dataset is pre-backed up, no additional backup will be created")
        print("=" * 60)
    else:
        print("=== EXECUTION MODE ===")
        print("Files will be ACTUALLY removed!")
        print("Dataset is pre-backed up, no additional backup will be created")
        print("=" * 60)
    
    if args.split == "all":
        splits = ["train", "test", "validation"]
    else:
        splits = [args.split]
    
    for split in splits:
        process_dataset_split(args.data_dir, split, args.dry_run)
    
    print("\nProcessing complete!")
    if args.dry_run:
        print("To actually remove files, run with --execute flag")


if __name__ == "__main__":
    main()