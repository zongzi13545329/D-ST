import os
import sys
import shutil

# Add project root to path to allow absolute imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import argparse
import numpy as np
import pandas as pd
import torch
import time

from src.datasets import (
    SIGF_Survival_Dataset, SIGF_Longitudinal_Survival_Dataset
)
from src.utils import set_seed, val_worker_init_fn, evaluate, evaluate_LTSA
from src.losses import CrossEntropySurvLoss, NLLSurvLoss, CoxSurvLoss
from src.models_long import create_model
from src.models_sf import create_model as sf_create_model

def SIGF_collate_fn(data):
    # x, y, censorship, rel_time, and obs_time are pre-padded in the Dataset.
    # We just need to stack them.
    x = torch.stack([d['x'] for d in data], dim=0).reshape(-1, 3, 224, 224)  # batch*max_seq_len x 3 x 224 x 224
    y = torch.stack([d['y'] for d in data], dim=0)  # batch x max_seq_len
    censorship = torch.stack([d['censorship'] for d in data], dim=0)  # batch x max_seq_len

    rel_time = torch.stack([d['rel_time'] for d in data], dim=0)  # batch x max_seq_len
    prior_AMD_sev = None  # SIGF doesn't have AMD severity data (like OHTS)
    obs_time = torch.stack([d['obs_time'] for d in data], dim=0)  # batch x max_seq_len

    # These are not padded, handle them as before
    fname = np.concatenate([d['fname'] for d in data])
    patient_id = np.concatenate([d['patient_id'] for d in data])
    laterality = np.concatenate([d['laterality'] for d in data])
    event_time = np.concatenate([d['event_time'] for d in data])

    seq_length = [d['seq_length'] for d in data]  # batch

    return {'x': x, 'y': y, 'censorship': censorship, 'event_time': event_time, 'obs_time': obs_time, 'fname': fname, 'seq_length': seq_length, 'rel_time': rel_time, 'prior_AMD_sev': prior_AMD_sev, 'patient_id': patient_id, 'laterality': laterality}

def SIGF_collate_fn_sf(data):
    # Get the maximum sequence length in this batch  
    max_seq_len = max([d['x'].shape[0] for d in data])
    
    # Pad all sequences to the same length and stack
    x_padded = []
    for d in data:
        x_seq = d['x']  # (seq_len, 3, 224, 224)
        if x_seq.shape[0] < max_seq_len:
            # Pad with zeros to max_seq_len
            pad_len = max_seq_len - x_seq.shape[0]
            x_padded_seq = torch.cat([x_seq, torch.zeros(pad_len, 3, 224, 224)], dim=0)
        else:
            x_padded_seq = x_seq
        x_padded.append(x_padded_seq)
    
    # Keep temporal structure: x is (B, T, C, H, W)
    x = torch.stack(x_padded, dim=0)  # (B, T, C, H, W)
    y = torch.stack([d['y'] for d in data], dim=0)  # (B, T)
    censorship = torch.stack([d['censorship'] for d in data], dim=0)  # (B, T)
    rel_time = torch.stack([d['rel_time'] for d in data], dim=0)  # (B, T)
    obs_time = torch.stack([d['obs_time'] for d in data], dim=0)  # (B, T)

    prior_AMD_sev = [d['prior_AMD_sev'] for d in data]  # Keep as None list for SIGF
    fname = np.concatenate([d['fname'] for d in data])  # sum(B*T)
    patient_id = np.concatenate([d['patient_id'] for d in data])  # sum(B*T)
    laterality = np.concatenate([d['laterality'] for d in data])  # sum(B*T)
    event_time = np.concatenate([d['event_time'] for d in data])  # sum(B*T)

    seq_length = [d['seq_length'] for d in data]  # (B,)

    return {
        'x': x,  # (B, T, C, H, W) for Timesformer
        'y': y,  # (B, T)
        'censorship': censorship,
        'event_time': event_time,
        'obs_time': obs_time,
        'fname': fname,
        'seq_length': seq_length,
        'rel_time': rel_time,
        'prior_AMD_sev': prior_AMD_sev,
        'patient_id': patient_id,
        'laterality': laterality
    }

def load_pretrained_weights(model, weight_path, device):
    """Load pretrained weights from OHTS training"""
    if os.path.exists(weight_path):
        checkpoint = torch.load(weight_path, map_location=device)
        if 'weights' in checkpoint:
            weights = checkpoint['weights']
        else:
            weights = checkpoint
        
        # Load weights
        missing_keys, unexpected_keys = model.load_state_dict(weights, strict=False)
        print(f"Loaded weights from {weight_path}")
        if missing_keys:
            print(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"Unexpected keys: {unexpected_keys}")
        return weights
    else:
        raise FileNotFoundError(f"Weight file not found: {weight_path}")

def main(args):
    # Set model name based on inference configuration
    MODEL_NAME = 'surv'
    MODEL_NAME += f'_SIGF'  # Using SIGF dataset for inference
    MODEL_NAME += f'_{args.model}'
    MODEL_NAME += f'_inference'  # Mark as inference
    MODEL_NAME += f'_from_OHTS'  # Mark source dataset
    
    # Create result directory in results/val
    val_results_dir = os.path.join(args.results_dir, 'val')
    if not os.path.isdir(val_results_dir):
        os.makedirs(val_results_dir)
    
    model_dir = os.path.join(val_results_dir, MODEL_NAME)
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # Create empty history CSV for compatibility with evaluate functions
    column_list = ['epoch', 'phase', 'loss'] + \
        [f'c_index_t-{t}_del-t-{del_t}' for t in args.t_list for del_t in args.del_t_list] + \
        [f'brier_t-{t}_del-t-{del_t}' for t in args.t_list for del_t in args.del_t_list] + \
        ['mean_c_index', 'mean_brier']
    history = pd.DataFrame(columns=column_list)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure SIGF dataset settings
    args.data_dir = '/home/lin01231/public/datasets/SIGF_processed'
    args.label_dir = '/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/datasets/SIGF_processed'
    args.n_classes = 15  # Max time_to_event = 14, so need 0-14 = 15 classes

    # Select appropriate dataset and collate function based on model type
    if args.model == 'LTSA':
        dataset = SIGF_Longitudinal_Survival_Dataset
        collate_fn = SIGF_collate_fn
        evaluate_fn = evaluate_LTSA
    elif args.model == 'SF':
        dataset = SIGF_Longitudinal_Survival_Dataset
        collate_fn = SIGF_collate_fn_sf
        evaluate_fn = evaluate_LTSA
    else:  # 'image' baseline
        dataset = SIGF_Survival_Dataset
        collate_fn = None
        evaluate_fn = evaluate

    # Create test dataset (we only need test set for inference)
    test_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='test', 
                          augment=False, tpe_mode=args.tpe_mode, learned_pe=args.learned_pe)

    # Create test data loader
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, 
                                            num_workers=4, shuffle=False, pin_memory=True, 
                                            worker_init_fn=val_worker_init_fn, collate_fn=collate_fn)

    # Create model
    if args.model == 'SF':
        model = sf_create_model(args).to(device)
    else:
        model = create_model(args).to(device)
    
    print(f"Model created: {args.model}")
    print(model)

    # Load pretrained weights from OHTS training
    weights = load_pretrained_weights(model, args.weight_path, device)

    # Get loss function (same as used in training)
    if args.loss == 'ce':
        loss_fn = CrossEntropySurvLoss(beta=args.beta)
    elif args.loss == 'nll':
        loss_fn = NLLSurvLoss(beta=args.beta)
    elif args.loss == 'cox':
        loss_fn = CoxSurvLoss()

    # Run inference evaluation
    print(f"Starting inference evaluation on SIGF dataset using {args.model} model...")
    start_time = time.time()
    
    evaluate_fn(model=model, device=device, loss_fn=loss_fn, data_loader=test_loader, 
               history=history, model_dir=model_dir, weights=weights, amp=args.amp, 
               t_list=args.t_list, del_t_list=args.del_t_list, dataset='SIGF', 
               step_ahead=args.step_ahead)

    # Report total inference time
    elapsed_time = time.time() - start_time
    print(f"[INFO] Total inference time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    # Save inference summary
    with open(os.path.join(model_dir, "inference_summary.txt"), "w") as f:
        f.write(f"Inference Configuration:\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Source weights: {args.weight_path}\n")
        f.write(f"Target dataset: SIGF\n")
        f.write(f"Total inference time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)\n")
        f.write(f"Test samples evaluated: {len(test_dataset)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference script for survival analysis models')
    
    # Required arguments
    parser.add_argument('--model', type=str, required=True, choices=['image', 'LTSA', 'SF'], 
                       help='Model type to use for inference')
    parser.add_argument('--weight_path', type=str, required=True,
                       help='Path to pretrained weights from OHTS training')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save inference results')
    
    # Model configuration (should match the training configuration)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--arch', type=str, default='resnet18', 
                       choices=['resnet18', 'swin_v2_t', 'convnext_t', 'caformer_s36', 'vit_base'])
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--attn_map', action='store_true', default=False, 
                       help='Whether to return attention maps for LTSA')
    parser.add_argument('--tpe', action='store_true', default=False)
    parser.add_argument('--tpe_mode', type=str, default='months', choices=['bins', 'months'])
    parser.add_argument('--amd_sev_enc', action='store_true', default=False,
                       help='Embed AMD severity score from prior visit')
    parser.add_argument('--learned_pe', action='store_true', default=False)
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'nll', 'cox'])
    parser.add_argument('--beta', type=float, default=0.15)
    parser.add_argument('--max_seq_len', type=int, default=14)
    parser.add_argument('--step_ahead', action='store_true', default=False)
    parser.add_argument('--use_deformable_spatial', action='store_true', default=False)
    parser.add_argument('--use_deformable_temporal', action='store_true', default=False)
    parser.add_argument('--mean_pool', action='store_true', default=False, 
                       help='Use mean pooling instead of [CLS] token')
    
    # VFM-specific arguments
    parser.add_argument('--vfm_checkpoint_path', type=str, 
                       default='/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/pretrained_weights/VFM_Fundus_weights.pth', 
                       help='Path to VisionFM pretrained weights')
    parser.add_argument('--vfm_patch_size', type=int, default=16, 
                       help='Patch size for VFM model (should match pretraining)')
    parser.add_argument('--vfm_checkpoint_key', type=str, default=None,
                       help='Key to use when loading the checkpoint (if checkpoint is a dict)')
    
    # Evaluation parameters
    parser.add_argument('--t_list', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    parser.add_argument('--del_t_list', type=int, nargs='+', default=[1, 2, 5, 8])

    args = parser.parse_args()
    
    print("Inference Configuration:")
    print(args)
    main(args)