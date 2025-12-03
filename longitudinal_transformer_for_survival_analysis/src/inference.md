## 推理文件构建计划 (Inference Script Planning)

### 分析总结
通过阅读train.py和utils.py，我理解了训练结构：

1. **训练流程**：数据加载 → 模型创建 → 训练循环 → 验证 → 测试评估
2. **测试部分在evaluate_LTSA/evaluate函数中**：加载最佳权重 → 模型推理 → 计算指标 → 保存结果
3. **结果输出**：训练历史图表、C-index/Brier表格、测试总结文本、预测结果pickle文件

### 推理脚本设计

#### 1. 核心思路
- **复用测试逻辑**：直接调用现有的evaluate_LTSA和evaluate函数
- **跳过训练**：不执行训练和验证循环，直接进行测试阶段
- **权重加载**：从OHTS训练好的权重文件中加载模型参数
- **数据切换**：将数据集从OHTS切换到SIGF进行测试

#### 2. 文件结构
创建`inference.py`文件，包含以下组件：
- 参数解析（复用train.py的argparse结构）
- 数据加载器设置（使用SIGF数据集）
- 模型创建和权重加载
- 测试评估（调用evaluate函数）
- 结果保存

#### 3. 三个模型的权重路径
根据训练命令，需要加载的权重文件：
- **Baseline**: `/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/src/results/surv_OHTS_image_50-ep_deform-spatial_deform-temporal/best.pt`
- **LTSA**: `/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/src/results/surv_OHTS_LTSA_step-ahead_50-ep_deform-spatial_deform-temporal/best.pt`  
- **SF**: `/home/lin01231/zhan9191/AI4M/SongProj/longitudinal_transformer_for_survival_analysis/src/results/surv_OHTS_SF_step-ahead_50-ep_deform-spatial_deform-temporal/best.pt`

#### 4. 主要修改点
1. **跳过训练循环**：直接进入测试阶段
2. **权重加载逻辑**：从指定路径加载OHTS训练的权重
3. **数据集配置**：使用SIGF数据集进行测试
4. **结果目录**：在results/val/下创建测试结果目录
5. **评估函数适配**：处理evaluate_LTSA中训练相关的参数

#### 5. 实现步骤
1. 创建inference.py脚本
2. 复制并修改train.py的数据加载和模型创建逻辑
3. 实现权重加载功能
4. 调用相应的evaluate函数
5. 确保结果输出格式与训练一致

#### 6. 预期输出
在`results/val/`目录下生成类似训练结果的文件结构：
- `surv_SIGF_image_inference/` (Baseline测试结果)
- `surv_SIGF_LTSA_inference/` (LTSA测试结果)  
- `surv_SIGF_SF_inference/` (SF测试结果)

每个目录包含：
- 各种指标的图表文件
- test_summary.txt (测试总结)
- test_preds.pkl (预测结果)
- test_c-index.csv / test_brier.csv (指标表格)