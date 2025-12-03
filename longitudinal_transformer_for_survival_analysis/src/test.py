import argparse
import os
import random
import torch
import numpy as np
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image
from models_sf import create_model as sf_create_model
from models_long import create_model as long_create_model
from utils import set_seed
from matplotlib import pyplot as plt
import importlib.util
import sys

# 强制从当前目录加载 datasets.py
spec = importlib.util.spec_from_file_location("datasets", "/home/lin01231/song0760/longitudinal_transformer_for_survival_analysis/src/datasets.py")
datasets = importlib.util.module_from_spec(spec)
sys.modules["datasets"] = datasets
spec.loader.exec_module(datasets)

AREDS_Longitudinal_Survival_Dataset = datasets.AREDS_Longitudinal_Survival_Dataset
OHTS_Longitudinal_Survival_Dataset = datasets.OHTS_Longitudinal_Survival_Dataset
SIGF_Longitudinal_Survival_Dataset = datasets.SIGF_Longitudinal_Survival_Dataset

def overlay_attention_on_fundus(fundus_tensor, attention_map, save_path, alpha=0.5):
    fundus = transforms.ToPILImage()(fundus_tensor.cpu()).convert("RGB").resize((224, 224))
    attn = attention_map
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    attn = Image.fromarray(np.uint8(attn * 255)).resize((224, 224), resample=Image.Resampling.BILINEAR)
    attn = np.array(attn)
    cmap = plt.get_cmap('jet')
    attn_color = cmap(attn / 255.0)[..., :3]
    fundus_np = np.array(fundus) / 255.0
    overlay = (1 - alpha) * fundus_np + alpha * attn_color
    overlay_img = Image.fromarray(np.uint8(overlay * 255))
    overlay_img.save(save_path)

def test_attention_visualization(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'AREDS':
        args.data_dir = '/home/lin01231/public/datasets/AMD_224/AMD_224'
        args.label_dir = '/home/lin01231/song0760/longitudinal_transformer_for_survival_analysis/datasets/AREDS'
        dataset = AREDS_Longitudinal_Survival_Dataset
        from train import AREDS_collate_fn_sf as collate_fn
        args.n_classes = 27
    elif args.dataset == 'OHTS':
        args.data_dir = '/home/lin01231/public/datasets/image_crop2/image_crop2'
        args.label_dir = '/home/lin01231/song0760/longitudinal_transformer_for_survival_analysis/datasets/OHTS'
        dataset = OHTS_Longitudinal_Survival_Dataset
        from train import OHTS_collate_fn_sf as collate_fn
        args.n_classes = 15
    elif args.dataset == 'SIGF':
        args.data_dir = '/home/lin01231/public/datasets/SIGF_processed'
        args.label_dir = '/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/datasets/SIGF_processed'
        dataset = SIGF_Longitudinal_Survival_Dataset
        from train import SIGF_collate_fn_sf as collate_fn
        args.n_classes = 15 # Max time_to_event = 14, so need 0-14 = 15 classes

    # Create model
    if args.model == 'SF':
        model = sf_create_model(args).to(device)
    elif args.model == 'LTSA':
        model = long_create_model(args).to(device)
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    test_dataset = dataset(
        data_dir=args.data_dir,
        label_dir=args.label_dir,
        split='test',
        augment=False,
        tpe_mode=args.tpe_mode,
        learned_pe=args.learned_pe
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    checkpoint = torch.load(args.model_weights, map_location=device)
    if 'weights' in checkpoint:
        model.load_state_dict(checkpoint['weights'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # Append model name to output directory
    args.output_dir = os.path.join(args.output_dir, args.model.upper())
    os.makedirs(args.output_dir, exist_ok=True)

    count = 0
    with torch.no_grad():
        for batch in test_loader:
            x = batch['x'].to(device)
            rel_times = batch['rel_time'].to(device)
            seq_lengths = batch['seq_length']
            prior_AMD_sevs = batch['prior_AMD_sev'] if 'prior_AMD_sev' in batch else None
            B, T, C, H, W = x.shape

            if args.model == 'SF':
                if args.attn_mode == 'spatial':
                    feat = model.encoder.embed(x.view(B * T, C, H, W))
                    attn_map = feat.mean(1).view(B, T, 14, 14)[0, T // 2]
                    img = x[0, T // 2]
                    valid_idx = min(T // 2, len(batch['patient_id']) - 1,len(batch['laterality']) - 1)
                    fname = f"{batch['patient_id'][valid_idx]}_{batch['laterality'][valid_idx]}_spatial_attn.png"
                    save_path = os.path.join(args.output_dir, fname)
                    overlay_attention_on_fundus(img, attn_map.cpu().numpy(), save_path)

                elif args.attn_mode == 'temporal':
                    encoded = model.encoder(x, seq_lengths)
                    encoded = model.pos_encoder(encoded, rel_times)
                    if model.temporal_encoder:
                        out = model.temporal_encoder(encoded)
                        attn_map = out.norm(dim=2)[0]
                        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                        img = x[0, T // 2]  # 中间帧
                        valid_idx = min(T // 2, len(batch['patient_id']) - 1,len(batch['laterality']) - 1)
                        fname = f"{batch['patient_id'][valid_idx]}_{batch['laterality'][valid_idx]}_temporal_attn.png"
                        save_path = os.path.join(args.output_dir, fname)
                        overlay_attention_on_fundus(img, attn_map.cpu().numpy(), save_path)

            elif args.model == 'LTSA':
                output = model(x.reshape(-1, 3, 224, 224), seq_lengths, rel_times, prior_AMD_sevs)
                if args.step_ahead and args.attn_map:
                    _, _, _, _, _, attn_map = output
                elif args.attn_map:
                    _, _, _, attn_map = output
                else:
                    raise ValueError("Attention map visualization requires --attn_map")

                attn_map = attn_map[0].mean(0).cpu().numpy()
                attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
                img = x[0, T // 2]
                valid_idx = min(T // 2, len(batch['patient_id']) - 1, len(batch['laterality']) - 1)
                fname = f"{batch['patient_id'][valid_idx]}_{batch['laterality'][valid_idx]}_ltsa_attn.png"
                save_path = os.path.join(args.output_dir, fname)
                overlay_attention_on_fundus(img, attn_map, save_path)

            count += 1
            if count >= args.num_samples:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/lin01231/public/datasets/AMD_224/AMD_224')
    parser.add_argument('--label_dir', type=str, default='/prj0129/grh4006/AREDS/labels')
    parser.add_argument('--dataset', type=str, default='AREDS', choices=['AREDS', 'OHTS', 'SIGF'])
    parser.add_argument('--model_weights', type=str, default='/home/lin01231/song0760/longitudinal_transformer_for_survival_analysis/src/results/surv_AREDS_SF_step-ahead_50-ep_deform-spatial_deform-temporal/best.pt')
    parser.add_argument('--output_dir', type=str, default='results/attn_viz')
    parser.add_argument('--model', type=str, default='SF', choices=['SF', 'LTSA'])
    parser.add_argument('--attn_mode', type=str, default='spatial', choices=['spatial', 'temporal'])
    parser.add_argument('--use_deformable_spatial', action='store_true',default=True )
    parser.add_argument('--use_deformable_temporal', action='store_true',default=True)
    parser.add_argument('--tpe_mode', type=str, default='months')
    parser.add_argument('--learned_pe', action='store_true')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--attn_map', action='store_true')
    parser.add_argument('--step_ahead', action='store_true')
    parser.add_argument('--arch', type=str, default='resnet18')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--n_classes', type=int, default=15)
    parser.add_argument('--mean_pool', action='store_true', default=False)
    parser.add_argument('--max_seq_len', type=int, default=14)
    parser.add_argument('--tpe', action='store_true', default=False)
    parser.add_argument('--amd_sev_enc', action='store_true', default=False, help='Embed AMD severity score from prior visit')

    args = parser.parse_args()
    test_attention_visualization(args)
