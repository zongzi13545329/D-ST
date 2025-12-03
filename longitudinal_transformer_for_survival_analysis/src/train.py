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
    AREDS_Survival_Dataset, OHTS_Survival_Dataset, SIGF_Survival_Dataset,
    AREDS_Longitudinal_Survival_Dataset, OHTS_Longitudinal_Survival_Dataset, SIGF_Longitudinal_Survival_Dataset,
    VFM_AREDS_Survival_Dataset, VFM_OHTS_Survival_Dataset, VFM_SIGF_Survival_Dataset,
    VFM_AREDS_Longitudinal_Survival_Dataset, VFM_OHTS_Longitudinal_Survival_Dataset, VFM_SIGF_Longitudinal_Survival_Dataset
)
from src.utils import set_seed, worker_init_fn, val_worker_init_fn, train, validate, evaluate, train_LTSA, validate_LTSA, evaluate_LTSA
from src.losses import CrossEntropySurvLoss, NLLSurvLoss, CoxSurvLoss
from src.models_long import create_model
from src.models_sf import create_model as sf_create_model

def AREDS_collate_fn(data):
    x = torch.stack([d['x'] for d in data], dim=0).reshape(-1, 3, 224, 224)  # batch*seq_length x 3 x 224 x 224
    y = torch.stack([d['y'] for d in data], dim=0)  # batch x 1
    censorship = torch.stack([d['censorship'] for d in data], dim=0)  # batch x 1

    rel_time = torch.stack([d['rel_time'] for d in data], dim=0)  # batch x seq_length
    prior_AMD_sev = torch.stack([d['prior_AMD_sev'] for d in data], dim=0)  # batch x seq_length
    obs_time = torch.stack([d['obs_time'] for d in data], dim=0)  # batch x seq_length

    fname = np.concatenate([d['fname'] for d in data])  # sum_batch_seq_len
    patient_id = np.concatenate([d['patient_id'] for d in data])  # sum_batch_seq_len
    laterality = np.concatenate([d['laterality'] for d in data])  # sum_batch_seq_len
    event_time = np.concatenate([d['event_time'] for d in data])  # sum_batch_seq_len

    seq_length = [d['seq_length'] for d in data]  # batch

    return {'x': x, 'y': y, 'censorship': censorship, 'event_time': event_time, 'obs_time': obs_time, 'fname': fname, 'seq_length': seq_length, 'rel_time': rel_time, 'prior_AMD_sev': prior_AMD_sev, 'patient_id': patient_id, 'laterality': laterality}

def AREDS_collate_fn_sf(data):
    # 保持时间结构：x 是 (B, T, C, H, W)
    x = torch.stack([d['x'] for d in data], dim=0)  # (B, T, C, H, W)
    y = torch.stack([d['y'] for d in data], dim=0)  # (B, T)
    censorship = torch.stack([d['censorship'] for d in data], dim=0)  # (B, T)
    rel_time = torch.stack([d['rel_time'] for d in data], dim=0)  # (B, T)
    obs_time = torch.stack([d['obs_time'] for d in data], dim=0)  # (B, T)
    prior_AMD_sev = torch.stack([d['prior_AMD_sev'] for d in data], dim=0)  # (B, T)

    fname = np.concatenate([d['fname'] for d in data])  # (B*T,)
    patient_id = np.concatenate([d['patient_id'] for d in data])  # (B*T,)
    laterality = np.concatenate([d['laterality'] for d in data])  # (B*T,)
    event_time = np.concatenate([d['event_time'] for d in data])  # (B*T,)
    seq_length = [d['seq_length'] for d in data]  # (B,)

    return {
        'x': x,  # (B, T, C, H, W)
        'y': y,  # (B, T)
        'censorship': censorship,  # (B, T)
        'event_time': event_time,  # (B*T,)
        'obs_time': obs_time,      # (B, T)
        'fname': fname,
        'seq_length': seq_length,
        'rel_time': rel_time,      # (B, T)
        'prior_AMD_sev': prior_AMD_sev,  # (B, T)
        'patient_id': patient_id,
        'laterality': laterality
    }

def OHTS_collate_fn(data):
    x = torch.stack([d['x'] for d in data], dim=0).reshape(-1, 3, 224, 224)  # batch*seq_length x 3 x 224 x 224
    y = torch.stack([d['y'] for d in data], dim=0)  # batch x 1
    censorship = torch.stack([d['censorship'] for d in data], dim=0)  # batch x 1

    rel_time = torch.stack([d['rel_time'] for d in data], dim=0)  # batch x seq_length
    prior_AMD_sev = None
    obs_time = torch.stack([d['obs_time'] for d in data], dim=0)  # batch x seq_length

    fname = np.concatenate([d['fname'] for d in data])  # sum_batch_seq_len
    patient_id = np.concatenate([d['patient_id'] for d in data])  # sum_batch_seq_len
    laterality = np.concatenate([d['laterality'] for d in data])  # sum_batch_seq_len
    event_time = np.concatenate([d['event_time'] for d in data])  # sum_batch_seq_len

    seq_length = [d['seq_length'] for d in data]  # batch

    return {'x': x, 'y': y, 'censorship': censorship, 'event_time': event_time, 'obs_time': obs_time, 'fname': fname, 'seq_length': seq_length, 'rel_time': rel_time, 'prior_AMD_sev': prior_AMD_sev, 'patient_id': patient_id, 'laterality': laterality}

def OHTS_collate_fn_sf(data):
    # 每个 sample['x'] 是 (T, C, H, W)，拼接为 (B, T, C, H, W)
    x = torch.stack([d['x'] for d in data], dim=0)  # 修正：保持时间结构，不再 reshape
    y = torch.stack([d['y'] for d in data], dim=0)  # (B, T)
    censorship = torch.stack([d['censorship'] for d in data], dim=0)  # (B, T)
    rel_time = torch.stack([d['rel_time'] for d in data], dim=0)  # (B, T)
    obs_time = torch.stack([d['obs_time'] for d in data], dim=0)  # (B, T)

    prior_AMD_sev = [d['prior_AMD_sev'] for d in data]  # 保持结构
    fname = np.concatenate([d['fname'] for d in data])  # sum(B*T)
    patient_id = np.concatenate([d['patient_id'] for d in data])  # sum(B*T)
    laterality = np.concatenate([d['laterality'] for d in data])  # sum(B*T)
    event_time = np.concatenate([d['event_time'] for d in data])  # sum(B*T)

    seq_length = [d['seq_length'] for d in data]  # (B,)

    return {
        'x': x,  # (B, T, C, H, W) ✅ 用于 Timesformer
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

def count_parameters(model):
    """
    Count the total number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    # Set detailed model name
    MODEL_NAME = 'surv'
    MODEL_NAME += f'_{args.dataset}'
    # MODEL_NAME += f'_{args.arch}'
    MODEL_NAME += f'_{args.model}'
    # MODEL_NAME += f'_{args.loss}'
    # MODEL_NAME += f'_lr-reduce-x{args.reduce_lr_factor}-p{args.reduce_lr_patience}' if args.reduce_lr else ''
    # MODEL_NAME += f'_val-{args.val}'
    # MODEL_NAME += f'_beta-{args.beta}' if args.loss != 'cox' else ''
    MODEL_NAME += f'_step-ahead' if args.step_ahead else ''
    MODEL_NAME += 'learned-pe' if args.learned_pe else ''
    # MODEL_NAME += f'_tpe-{args.tpe_mode}' if args.tpe else ''
    # MODEL_NAME += '_AMD-sev-enc' if args.amd_sev_enc else ''
    # MODEL_NAME += f'_{args.n_layers}-layers' if args.n_layers > 1 else ''
    # MODEL_NAME += f'_{args.n_heads}-heads' if args.model == 'transformer' else ''
    # MODEL_NAME += f'_bs-{args.batch_size}'
    # MODEL_NAME += '_amp' if args.amp else ''
    # MODEL_NAME += f'_lr-{args.lr}'
    MODEL_NAME += f'_{args.max_epochs}-ep'
    # MODEL_NAME += f'_patience-{args.patience}'
    # MODEL_NAME += '_aug' if args.augment else ''
    # MODEL_NAME += f'_drp-{args.dropout}' if args.dropout > 0 else ''
    MODEL_NAME += f'_seed-{args.seed}' if args.seed != 0 else ''
    MODEL_NAME += '_deform-spatial' if args.use_deformable_spatial else '_no-deform-spatial'
    MODEL_NAME += '_deform-temporal' if args.use_deformable_temporal else '_no-deform-temporal'
    # Add VFM-specific naming components if using a VFM model
    if args.model in ['VFM_image', 'VFM_LTSA']:
        MODEL_NAME += f'_vfm-{args.vfm_patch_size}'

    # Create result directory and model directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)
    model_dir = os.path.join(args.results_dir, MODEL_NAME)

    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)
    os.mkdir(model_dir)

    # Create csv to log training history
    column_list = ['epoch', 'phase', 'loss'] + \
        [f'c_index_t-{t}_del-t-{del_t}' for t in args.t_list for del_t in args.del_t_list] + \
        [f'brier_t-{t}_del-t-{del_t}' for t in args.t_list for del_t in args.del_t_list] + \
        ['mean_c_index', 'mean_brier']
    history = pd.DataFrame(columns=column_list)
    history.to_csv(os.path.join(model_dir, 'history.csv'), index=False)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'AREDS':
        args.data_dir = '/home/lin01231/public/datasets/AMD_224/AMD_224'
        args.label_dir = '/home/lin01231/song0760/longitudinal_transformer_for_survival_analysis/datasets/AREDS'
        args.n_classes = 27  # 6-month intervals for 13 years

        if args.model == 'LTSA':
            dataset = AREDS_Longitudinal_Survival_Dataset
            collate_fn = AREDS_collate_fn
        elif args.model == 'VFM_LTSA':  # Added VFM_LTSA option
            dataset = VFM_AREDS_Longitudinal_Survival_Dataset
            collate_fn = AREDS_collate_fn
        elif args.model == 'SF':
            dataset = AREDS_Longitudinal_Survival_Dataset
            collate_fn = AREDS_collate_fn_sf
        elif args.model == 'VFM_image':
            dataset = VFM_AREDS_Survival_Dataset
            collate_fn = None
        else:  # 'image'
            dataset = AREDS_Survival_Dataset
            collate_fn = None
    elif args.dataset == 'OHTS':
        args.data_dir = '/home/lin01231/public/datasets/image_crop2/image_crop2'
        args.label_dir = '/home/lin01231/song0760/longitudinal_transformer_for_survival_analysis/datasets/OHTS'
        args.n_classes = 15  # 1-year intervals for 14 years

        if args.model == 'LTSA':
            dataset = OHTS_Longitudinal_Survival_Dataset
            collate_fn = OHTS_collate_fn
        elif args.model == 'VFM_LTSA':  # Added VFM_LTSA option
            dataset = VFM_OHTS_Longitudinal_Survival_Dataset
            collate_fn = OHTS_collate_fn
        elif args.model == 'SF':
            dataset = OHTS_Longitudinal_Survival_Dataset
            collate_fn = OHTS_collate_fn_sf
        elif args.model == 'VFM_image':
            dataset = VFM_OHTS_Survival_Dataset
            collate_fn = None
        else:  # 'image'
            dataset = OHTS_Survival_Dataset
            collate_fn = None
    elif args.dataset == 'SIGF':
        args.data_dir = '/home/lin01231/public/datasets/SIGF_processed'
        args.label_dir = '/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/datasets/SIGF_processed'
        args.n_classes = 15  # Max time_to_event = 14, so need 0-14 = 15 classes

        if args.model == 'LTSA':
            dataset = SIGF_Longitudinal_Survival_Dataset
            collate_fn = SIGF_collate_fn
        elif args.model == 'VFM_LTSA':  # VFM_LTSA option for SIGF
            dataset = VFM_SIGF_Longitudinal_Survival_Dataset
            collate_fn = SIGF_collate_fn
        elif args.model == 'SF':
            dataset = SIGF_Longitudinal_Survival_Dataset
            collate_fn = SIGF_collate_fn_sf
        elif args.model == 'VFM_image':
            dataset = VFM_SIGF_Survival_Dataset
            collate_fn = None
        else:  # 'image'
            dataset = SIGF_Survival_Dataset
            collate_fn = None

    # Create train, val, and test datasets
    train_dataset = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='train', augment=args.augment, tpe_mode=args.tpe_mode, learned_pe=args.learned_pe)
    val_dataset   = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='val', augment=False, tpe_mode=args.tpe_mode, learned_pe=args.learned_pe)
    test_dataset  = dataset(data_dir=args.data_dir, label_dir=args.label_dir, split='test', augment=False, tpe_mode=args.tpe_mode, learned_pe=args.learned_pe)

    # Create train, val, and test data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True, pin_memory=False, worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    val_loader   = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True, worker_init_fn=val_worker_init_fn, collate_fn=collate_fn)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False, pin_memory=True, worker_init_fn=val_worker_init_fn, collate_fn=collate_fn)

    # Create model
    if args.model == 'SF':
        model = sf_create_model(args).to(device)
    else:
        model = create_model(args).to(device)
    print(model)

    # Print total trainable parameters
    num_params = count_parameters(model)
    print(f"Total trainable parameters: {num_params:,}")

    # Get loss function
    if args.loss == 'ce':
        loss_fn = CrossEntropySurvLoss(beta=args.beta)
    elif args.loss == 'nll':
        loss_fn = NLLSurvLoss(beta=args.beta)
    elif args.loss == 'cox':
        loss_fn = CoxSurvLoss()

    # Get optimizer and learning rate scheduler (if enabled)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = None
    if args.reduce_lr:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max' if args.val == 'c-index' else 'min', factor=args.reduce_lr_factor, patience=args.reduce_lr_patience, verbose=True)

    if args.model in ['LTSA', 'SF']:
        train_fn, validate_fn, evaluate_fn = train_LTSA, validate_LTSA, evaluate_LTSA
    else:
        train_fn, validate_fn, evaluate_fn = train, validate, evaluate



    # Train with early stopping
    epoch = 1
    if args.val == 'c-index':
        early_stopping_dict = {'best_metric': 0., 'epochs_no_improve': 0, 'metric': args.val}
    elif args.val == 'loss':
        early_stopping_dict = {'best_metric': 1e8, 'epochs_no_improve': 0, 'metric': args.val}
    elif args.val == 'brier':
        early_stopping_dict = {'best_metric': 1e8, 'epochs_no_improve': 0, 'metric': args.val}
    best_model_wts = None
    start_time = time.time()
    while epoch <= args.max_epochs and early_stopping_dict['epochs_no_improve'] < args.patience:
        history = train_fn(model=model, device=device, loss_fn=loss_fn, optimizer=optimizer, data_loader=train_loader, history=history, epoch=epoch, model_dir=model_dir, amp=args.amp, t_list=args.t_list, del_t_list=args.del_t_list, dataset=args.dataset, step_ahead=args.step_ahead)
        history, early_stopping_dict, best_model_wts = validate_fn(model=model, device=device, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, data_loader=val_loader, history=history, epoch=epoch, model_dir=model_dir, early_stopping_dict=early_stopping_dict, best_model_wts=best_model_wts, amp=args.amp, t_list=args.t_list, del_t_list=args.del_t_list, dataset=args.dataset, step_ahead=args.step_ahead)

        epoch += 1

    # Evaluate trained model on test set
    evaluate_fn(model=model, device=device, loss_fn=loss_fn, data_loader=test_loader, history=history, model_dir=model_dir, weights=best_model_wts, amp=args.amp, t_list=args.t_list, del_t_list=args.del_t_list, dataset=args.dataset, step_ahead=args.step_ahead)

    # Report total training time
    elapsed_time = time.time() - start_time
    print(f"[INFO] Total training time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)")

    with open(os.path.join(model_dir, "training_summary.txt"), "w") as f:
        f.write(f"Model: {args.model}\n")
        f.write(f"Total trainable parameters: {num_params:,}\n")
        f.write(f"Total training time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)\n")
        f.write(f"Valid training samples used: {train_dataset.valid_sample_count}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/prj0129/grh4006/AMD_224')
    parser.add_argument('--label_dir', type=str, default='/prj0129/grh4006/AREDS/labels')
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32, help='Note that effective batch size is max_seq_len*batch_size when model=LTSA')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--arch', type=str, default='resnet18', choices=['resnet18', 'swin_v2_t', 'convnext_t', 'caformer_s36', 'vit_base'], help='Image encoder architecture')
    parser.add_argument('--model', type=str, default='image', choices=['image', 'LTSA', 'SF', 'VFM_image', 'VFM_LTSA'], help='Model type: Single-image baseline, Longitudinal Transformer, SF, or VFM variants')
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--attn_map', action='store_true', default=False, help='Whether to return attention maps for LTSA')
    parser.add_argument('--tpe', action='store_true', default=False, help='Use temporal positional encoding (TPE) to embed knowledge of visit time in longitudinal image sequences')
    parser.add_argument('--tpe_mode', type=str, default='months', choices=['bins', 'months'], help='Embed visit time measured in months or discrete 6-month time bins')
    parser.add_argument('--amd_sev_enc', action='store_true', default=False, help='Embed AMD severity score from prior visit')
    parser.add_argument('--learned_pe', action='store_true', default=False, help='Use learned positional encoding rather than fixed sinusoidal encoding')
    parser.add_argument('--loss', type=str, default='ce', choices=['ce', 'nll', 'cox'], help='Survival loss function (cross-entropy, negative log likelihood, or Cox)')
    parser.add_argument('--beta', type=float, default=0.15, help='Weight applied to term that upweights uncensored cases')
    parser.add_argument('--val', type=str, default='c-index', choices=['loss', 'c-index', 'brier'], help='Validation metric')
    parser.add_argument('--max_seq_len', type=int, default=14, help='Maximum sequence length (all sequences are padded to this length)')
    parser.add_argument('--t_list', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    parser.add_argument('--del_t_list', type=int, nargs='+', default=[1, 2, 5, 8])
    parser.add_argument('--step_ahead', action='store_true', default=False, help='Whether to enable step-ahead feature prediction')
    parser.add_argument('--dataset', type=str, default='AREDS', choices=['AREDS', 'OHTS', 'SIGF'])
    parser.add_argument('--reduce_lr', action='store_true', default=False, help='Whether to apply a "reduce on plataeu" learning rate scheduler')
    parser.add_argument('--reduce_lr_factor', type=float, default=0.5, help='Factor by which to reduce learning rate on plateau')
    parser.add_argument('--reduce_lr_patience', type=int, default=2, help='Patience for learning rate scheduler')
    parser.add_argument('--mean_pool', action='store_true', default=False, help='Use mean pooling instead of [CLS] token')
    parser.add_argument('--use_deformable_spatial', action='store_true', help='Use deformable attention in spatial encoder')
    parser.add_argument('--use_deformable_temporal', action='store_true', help='Use deformable attention for temporal encoder')
    
    # VFM-specific arguments (from VisionFM/finetune_visionfm_for_multiclass_classification.py)
    parser.add_argument('--vfm_checkpoint_path', type=str, default='/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/pretrained_weights/VFM_Fundus_weights.pth', 
                        help='Path to VisionFM pretrained weights')
    parser.add_argument('--vfm_patch_size', type=int, default=16, 
                        help='Patch size for VFM model (should match pretraining)')
    parser.add_argument('--vfm_checkpoint_key', type=str, default=None,
                        help='Key to use when loading the checkpoint (if checkpoint is a dict)')
 
    args = parser.parse_args()

    print(args)
    main(args)