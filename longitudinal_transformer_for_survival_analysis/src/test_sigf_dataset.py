#!/usr/bin/env python3
"""
Test script to verify SIGF dataset implementation
"""
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add project root to path for imports to allow for absolute imports (e.g., from src.utils)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from datasets import SIGF_Survival_Dataset, SIGF_Longitudinal_Survival_Dataset

def test_sigf_single_dataset():
    """Test SIGF single image dataset"""
    print("Testing SIGF_Survival_Dataset...")
    
    data_dir = '/home/lin01231/public/datasets/SIGF_processed'
    label_dir = '/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/datasets/SIGF_processed'
    
    # Test train split
    try:
        dataset = SIGF_Survival_Dataset(
            data_dir=data_dir,
            label_dir=label_dir,
            split='train',
            augment=False,
            tpe_mode='bins',
            learned_pe=False
        )
        
        print(f"✓ Train dataset loaded successfully. Length: {len(dataset)}")
        
        # Test loading a few samples
        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]
                print(f"  Sample {i}:")
                print(f"    x shape: {sample['x'].shape}")
                print(f"    y shape: {sample['y'].shape}")
                print(f"    patient_id: {sample['patient_id']}")
                print(f"    laterality: {sample['laterality']}")
                print(f"    label: {sample['y'].item()}")
                print(f"    event_time: {sample['event_time']}")
                print(f"    censorship: {sample['censorship'].item()}")
                break  # Only test first successful sample
            except Exception as e:
                print(f"  ✗ Failed to load sample {i}: {e}")
                continue
                
    except Exception as e:
        print(f"✗ Failed to create train dataset: {e}")

def test_sigf_longitudinal_dataset():
    """Test SIGF longitudinal dataset"""
    print("\nTesting SIGF_Longitudinal_Survival_Dataset...")
    
    data_dir = '/home/lin01231/public/datasets/SIGF_processed'
    label_dir = '/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/datasets/SIGF_processed'
    
    try:
        dataset = SIGF_Longitudinal_Survival_Dataset(
            data_dir=data_dir,
            label_dir=label_dir,
            split='train',
            augment=False,
            tpe_mode='bins',
            learned_pe=False
        )
        
        print(f"✓ Longitudinal dataset loaded successfully. Length: {len(dataset)}")
        
        # Test loading a sample
        for i in range(min(3, len(dataset))):
            try:
                sample = dataset[i]
                print(f"  Sample {i}:")
                print(f"    x shape: {sample['x'].shape}")
                print(f"    y shape: {sample['y'].shape}")
                print(f"    seq_length: {sample['seq_length']}")
                print(f"    patient_id[0]: {sample['patient_id'][0]}")
                print(f"    laterality[0]: {sample['laterality'][0]}")
                print(f"    rel_time shape: {sample['rel_time'].shape}")
                print(f"    prior_AMD_sev: {sample['prior_AMD_sev']}")
                break  # Only test first successful sample
            except Exception as e:
                print(f"  ✗ Failed to load sample {i}: {e}")
                continue
                
    except Exception as e:
        print(f"✗ Failed to create longitudinal dataset: {e}")

def test_sigf_collate_functions():
    """Test SIGF collate functions with DataLoader"""
    print("\nTesting SIGF collate functions...")
    
    data_dir = '/home/lin01231/public/datasets/SIGF_processed'
    label_dir = '/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/datasets/SIGF_processed'
    
    try:
        from train import SIGF_collate_fn, SIGF_collate_fn_sf
        
        dataset = SIGF_Longitudinal_Survival_Dataset(
            data_dir=data_dir,
            label_dir=label_dir,
            split='train',
            augment=False,
            tpe_mode='bins',
            learned_pe=False
        )
        
        # Test LTSA collate function
        loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=SIGF_collate_fn)
        batch = next(iter(loader))
        
        print("✓ SIGF_collate_fn test:")
        print(f"  x shape: {batch['x'].shape}")
        print(f"  y shape: {batch['y'].shape}")
        print(f"  seq_length: {batch['seq_length']}")
        print(f"  prior_AMD_sev: {batch['prior_AMD_sev']}")
        
        # Test SF collate function
        loader_sf = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=SIGF_collate_fn_sf)
        batch_sf = next(iter(loader_sf))
        
        print("✓ SIGF_collate_fn_sf test:")
        print(f"  x shape: {batch_sf['x'].shape}")
        print(f"  y shape: {batch_sf['y'].shape}")
        print(f"  seq_length: {batch_sf['seq_length']}")
        
    except Exception as e:
        print(f"✗ Failed to test collate functions: {e}")

if __name__ == "__main__":
    print("SIGF Dataset Implementation Test")
    print("=" * 40)
    
    test_sigf_single_dataset()
    test_sigf_longitudinal_dataset()
    test_sigf_collate_functions()
    
    print("\n" + "=" * 40)
    print("Test completed!")