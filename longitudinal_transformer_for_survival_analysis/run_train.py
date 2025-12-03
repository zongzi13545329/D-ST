# 这个脚本用于从项目根目录运行 train.py

import sys
import os

# 将当前目录添加到 Python 路径
sys.path.insert(0, os.path.abspath('.'))

# 导入 train.py 中的 main 函数
from src.train import main

if __name__ == "__main__":
    # 导入 argparse 以处理命令行参数
    import argparse
    
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='Run training script')
    
    # 添加与 train.py 相同的参数
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
    parser.add_argument('--dataset', type=str, default='AREDS', choices=['AREDS', 'OHTS'])
    parser.add_argument('--reduce_lr', action='store_true', default=False, help='Whether to apply a "reduce on plataeu" learning rate scheduler')
    parser.add_argument('--reduce_lr_factor', type=float, default=0.5, help='Factor by which to reduce learning rate on plateau')
    parser.add_argument('--reduce_lr_patience', type=int, default=2, help='Patience for learning rate scheduler')
    parser.add_argument('--mean_pool', action='store_true', default=False, help='Use mean pooling instead of [CLS] token')
    parser.add_argument('--use_deformable_spatial', action='store_true', help='Use deformable attention in spatial encoder')
    parser.add_argument('--use_deformable_temporal', action='store_true', help='Use deformable attention for temporal encoder')
    
    # VFM-specific arguments
    parser.add_argument('--vfm_checkpoint_path', type=str, default='/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/pretrained_weights/VFM_Fundus_weights.pth', 
                        help='Path to VisionFM pretrained weights')
    parser.add_argument('--vfm_patch_size', type=int, default=16, 
                        help='Patch size for VFM model (should match pretraining)')
    parser.add_argument('--vfm_checkpoint_key', type=str, default=None,
                        help='Key to use when loading the checkpoint (if checkpoint is a dict)')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 调用 main 函数
    main(args) 