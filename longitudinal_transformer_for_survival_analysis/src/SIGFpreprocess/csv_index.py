#!/usr/bin/env python3
"""
SIGF Dataset to CSV Converter (Index Version)
Converts SIGF processed dataset to CSV format with index-based year field.

For each patient-eye combination:
- ID: Patient ID (e.g., SD1294)
- LR: Eye designation (OD or OS)
- glaucoma: Always 0 (post-progression images removed)
- year: Index of image in sequence (0, 1, 2, ...)
- time_to_event: Total number of images for this patient-eye
- censorship: 0 if progression occurs (event), 1 if no progression (censored)
"""

import os
import csv
import re
from pathlib import Path
from collections import defaultdict


def parse_image_filename(filename):
    """Parse image filename to extract patient ID, date, and eye.
    
    Args:
        filename (str): Image filename (e.g., 'SD1284_1990_04_11_OS.JPG')
        
    Returns:
        tuple: (patient_id, date_str, eye) or None if parsing fails
    """
    # Remove file extension
    name = os.path.splitext(filename)[0]
    
    # Handle files with _01 suffix
    if name.endswith('_01'):
        name = name[:-3]
    
    # Pattern: SD{patient_id}_{yyyy}_{mm}_{dd}_{eye}
    pattern = r'SD(\d+)_(\d{4}_\d{2}_\d{2})_([OD|OS]+)$'
    match = re.match(pattern, name)
    
    if match:
        patient_id = match.group(1)
        date_str = match.group(2)
        eye = match.group(3)
        return f"SD{patient_id}", date_str, eye
    
    return None


def read_labels(label_file_path):
    """Read labels from text file.
    
    Args:
        label_file_path (str): Path to label file
        
    Returns:
        list: List of integer labels (0 or 1)
    """
    labels = []
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(int(line))
    except FileNotFoundError:
        print(f"Warning: Label file not found: {label_file_path}")
        return []
    except ValueError as e:
        print(f"Warning: Invalid label in {label_file_path}: {e}")
        return []
    
    return labels


def process_split(split_name, base_path, output_dir):
    """Process a single split (train/test/validation) and generate CSV.
    
    Args:
        split_name (str): Name of the split ('train', 'test', 'validation')
        base_path (str): Base path to SIGF_processed dataset
        output_dir (str): Output directory for CSV files
    """
    print(f"Processing {split_name} split...")
    
    # Set up paths based on split
    if split_name == 'train':
        image_path = os.path.join(base_path, split_name, 'image', 'all', 'image')
        label_path = os.path.join(base_path, split_name, 'label', 'label')
    else:
        image_path = os.path.join(base_path, split_name, 'image')
        label_path = os.path.join(base_path, split_name, 'label')
    
    # Check if paths exist
    if not os.path.exists(image_path):
        print(f"Error: Image path does not exist: {image_path}")
        return
    
    if not os.path.exists(label_path):
        print(f"Error: Label path does not exist: {label_path}")
        return
    
    # Collect all patient-eye combinations and their images
    patient_eye_data = defaultdict(list)
    
    # Process images
    if split_name == 'train':
        # Training set has patient-eye subdirectories
        for patient_eye_dir in os.listdir(image_path):
            patient_eye_path = os.path.join(image_path, patient_eye_dir)
            if os.path.isdir(patient_eye_path):
                images = []
                for img_file in os.listdir(patient_eye_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        parsed = parse_image_filename(img_file)
                        if parsed:
                            patient_id, date_str, eye = parsed
                            images.append((img_file, date_str, patient_id, eye))
                
                # Sort images by date
                images.sort(key=lambda x: x[1])
                patient_eye_data[patient_eye_dir] = images
    else:
        # Test/validation sets have patient-eye subdirectories
        for patient_eye_dir in os.listdir(image_path):
            patient_eye_path = os.path.join(image_path, patient_eye_dir)
            if os.path.isdir(patient_eye_path):
                images = []
                for img_file in os.listdir(patient_eye_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        parsed = parse_image_filename(img_file)
                        if parsed:
                            patient_id, date_str, eye = parsed
                            images.append((img_file, date_str, patient_id, eye))
                
                # Sort images by date
                images.sort(key=lambda x: x[1])
                patient_eye_data[patient_eye_dir] = images
    
    # Generate CSV data
    csv_data = []
    
    for patient_eye_dir, images in patient_eye_data.items():
        if not images:
            continue

        # New feature: If more than 14 images, skip the patient.
        if len(images) > 14:
            print(f"Skipping {patient_eye_dir}: found {len(images)} images, which is more than 14.")
            continue
        
        # Get patient ID and eye from first image
        _, _, patient_id, eye = images[0]
        
        # Read labels
        label_file = os.path.join(label_path, f"{patient_eye_dir}.txt")
        labels = read_labels(label_file)
        
        # SIGF_processed removes post-progression images but keeps all labels
        # We should only use images corresponding to label=0 (pre-progression)
        # Find the number of consecutive 0s from the beginning
        num_pre_progression = 0
        for label in labels:
            if label == 0:
                num_pre_progression += 1
            else:
                break  # Stop at first progression (label=1)
        
        if len(images) != num_pre_progression:
            print(f"Warning: Image count ({len(images)}) doesn't match pre-progression count ({num_pre_progression}) for {patient_eye_dir}")
            print(f"Labels: {labels}")
            continue
        
        # Determine censorship (0 if event occurred/disease progression, 1 if censored/no progression)
        # Standard survival analysis: censorship=0 means event happened, censorship=1 means censored
        censorship = 0 if any(label == 1 for label in labels) else 1
        time_to_event = len(images)
        
        # Generate CSV rows
        for idx, (img_file, date_str, _, _) in enumerate(images):
            csv_data.append({
                'ID': patient_id,
                'LR': eye,
                'glaucoma': 0,  # Always 0 as post-progression images are removed
                'year': idx,    # Index-based year
                'time_to_event': time_to_event,
                'censorship': censorship
            })
    
    # Write CSV file
    output_file = os.path.join(output_dir, f"{split_name}.csv")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'LR', 'glaucoma', 'year', 'time_to_event', 'censorship']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in csv_data:
            writer.writerow(row)
    
    print(f"Generated {output_file} with {len(csv_data)} rows")
    return len(csv_data)


def main():
    """Main function to process all splits."""
    # Paths
    base_path = "/home/lin01231/public/datasets/SIGF_processed"
    output_dir = "/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/datasets/SIGF_processed"
    
    # Process each split
    splits = ['train', 'test', 'validation']
    total_rows = 0
    
    for split in splits:
        try:
            rows = process_split(split, base_path, output_dir)
            if rows:
                total_rows += rows
        except Exception as e:
            print(f"Error processing {split}: {e}")
    
    print(f"\nProcessing complete. Total rows generated: {total_rows}")
    print(f"CSV files saved to: {output_dir}")


if __name__ == "__main__":
    main()