# SIGF Dataset Integration Debug Report

## Overview

本文档记录了将SIGF数据集集成到longitudinal transformer for survival analysis框架中的过程、遇到的问题以及当前状态。

## 已完成的工作

### 1. 代码文件修改

#### 1.1 datasets.py
- **新增类**：
  - `SIGF_Survival_Dataset` - 单图像生存分析数据集
  - `SIGF_Longitudinal_Survival_Dataset` - 纵向序列生存分析数据集
  - `VFM_SIGF_Survival_Dataset` - VFM版本单图像数据集
  - `VFM_SIGF_Longitudinal_Survival_Dataset` - VFM版本纵向数据集

- **关键实现特点**：
  - **DeepGF兼容预处理**：224×224 resize + 简单0-1归一化（无ImageNet-style normalization）
  - **无数据增强**：保持与DeepGF完全一致的预处理
  - **文件匹配逻辑**：处理`_01`后缀的图像文件
  - **路径映射**：正确处理`val`→`validation`目录名映射
  - **标签处理**：使用`time_to_event`作为分类标签（而非`year`索引）

#### 1.2 train.py
- **新增导入**：所有SIGF相关数据集类
- **新增collate函数**：
  - `SIGF_collate_fn` - LTSA模型用
  - `SIGF_collate_fn_sf` - SF模型用
- **数据集选择逻辑**：添加`--dataset SIGF`选项
- **路径配置**：
  - `data_dir`: `/home/lin01231/public/datasets/SIGF_processed`
  - `label_dir`: `/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/datasets/SIGF_processed`
  - `n_classes`: 21（基于max time_to_event = 20）

### 2. 路径问题修复

#### 2.1 图像文件匹配
- **问题**：原始逻辑无法匹配带`_01`后缀的图像文件
- **解决**：修改正则表达式匹配模式，支持`_{laterality}.JPG`和`_{laterality}_01.JPG`

#### 2.2 验证集路径映射
- **问题**：split='val'时路径构建为`/val/image/`，但实际目录为`/validation/image/`
- **解决**：在`_get_image_path`函数中添加路径映射逻辑

### 3. 数据集测试

创建了`test_sigf_dataset.py`测试脚本，验证结果：
- ✅ Single dataset: 2468个样本加载成功
- ✅ Longitudinal dataset: 295个患者眼部组合  
- ✅ 图像张量形状正确: (3, 224, 224)
- ✅ 序列张量形状正确: (14, 3, 224, 224)
- ✅ Collate函数工作正常



## 文件位置总结

### 主要代码文件
- `SongProj/longitudinal_transformer_for_survival_analysis/src/datasets.py` - 数据集类实现
- `SongProj/longitudinal_transformer_for_survival_analysis/src/train.py` - 训练脚本
- `SongProj/longitudinal_transformer_for_survival_analysis/src/test_sigf_dataset.py` - 测试脚本

### 数据文件
- 图像数据：`datasets/SIGF_processed/`
- CSV标签：`SongProj/longitudinal_transformer_for_survival_analysis/datasets/SIGF_processed/`
- 对比数据：`SongProj/longitudinal_transformer_for_survival_analysis/datasets/AREDS/` 和 `SongProj/longitudinal_transformer_for_survival_analysis/datasets/OHTS/`


### 文档文件
- `ongProj/longitudinal_transformer_for_survival_analysis/src/SIGFpreprocess/SIGF_Dataset_Analysis.md` - SIGF数据集分析
- `SongProj/longitudinal_transformer_for_survival_analysis/src/SIGFpreprocess/SIGF_integration.md` - 集成计划

## 下一步行动建议



## 命令参考

```bash
# 当前可用的训练命令
python train.py --results_dir results --dataset SIGF --model image --dropout 0.25 --augment --reduce_lr --batch_size 64
python train.py --results_dir results --dataset SIGF --model LTSA --dropout 0.25 --augment --reduce_lr --batch_size 32 --tpe --step_ahead  
python train.py --results_dir results --dataset SIGF --model SF --dropout 0.25 --augment --reduce_lr --batch_size 32 --tpe --step_ahead --use_deformable_spatial --use_deformable_temporal
```