# Deformable Spatiotemporal Attention for Efficient and Interpretable Longitudinal Survival Modeling

This is the official implementation code for the paper "Deformable Spatiotemporal Attention for Efficient and Interpretable Longitudinal Survival Modeling".

## Environment Setup

### 1. Install Deformable DETR

First, you need to install the Deformable DETR module. Please refer to the official README in the `deformable_detr` folder for detailed installation instructions.

### 2. Configure Environment

After successfully installing Deformable DETR, navigate to the `longitudinal_transformer_for_survival_analysis` folder and set up the environment using the provided conda configuration file:
```bash
cd longitudinal_transformer_for_survival_analysis
conda env create -f ltsa.yml
conda activate ltsa
```

## Dataset Preparation

### 1. Download Datasets

You need to apply for and download the AREDS and OHTS datasets through their official access procedures:
- **AREDS**: Age-Related Eye Disease Study
- **OHTS**: Ocular Hypertension Treatment Study

The expected data structure is:
```
your_path/AMD_224/patient_id/patient_image.jpg
```

Dataset split files are provided in the `datasets` folder.

### 2. Update Data Paths

Modify the dataset paths in `train.py` according to your local data location.

## Training

Use the following commands to train different models:

### Baseline Models

**Image-based baseline:**
```bash
python train.py --results_dir results --dataset OHTS --model image --dropout 0.25 --augment --reduce_lr --batch_size 64
```

**LTSA (LSTM-based):**
```bash
python train.py --results_dir results --dataset OHTS --model LTSA --dropout 0.25 --augment --reduce_lr --batch_size 32 --tpe --step_ahead
```

**SF (Survival Forest with Deformable Attention):**
```bash
python train.py --results_dir results --dataset OHTS --model SF --dropout 0.25 --augment --reduce_lr --batch_size 32 --tpe --step_ahead --use_deformable_spatial --use_deformable_temporal
```

### AREDS Dataset

**Image-based baseline:**
```bash
python train.py --results_dir results --dataset AREDS --model image --dropout 0.25 --augment --reduce_lr --batch_size 32
```

**LTSA:**
```bash
python train.py --results_dir results --dataset AREDS --model LTSA --dropout 0.25 --augment --reduce_lr --batch_size 32 --tpe --step_ahead
```

**SF:**
```bash
python train.py --results_dir results --dataset AREDS --model SF --dropout 0.25 --augment --reduce_lr --batch_size 32 --tpe --step_ahead
```

## Key Arguments

- `--results_dir`: Directory to save training results
- `--dataset`: Dataset name (OHTS or AREDS)
- `--model`: Model architecture (image, LTSA, or SF)
- `--dropout`: Dropout rate
- `--augment`: Enable data augmentation
- `--reduce_lr`: Enable learning rate reduction on plateau
- `--batch_size`: Training batch size
- `--tpe`: Enable temporal positional encoding
- `--step_ahead`: Enable step-ahead prediction
- `--use_deformable_spatial`: Use deformable spatial attention
- `--use_deformable_temporal`: Use deformable temporal attention
