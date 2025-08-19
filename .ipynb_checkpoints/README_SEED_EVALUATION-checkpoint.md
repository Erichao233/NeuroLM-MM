# SEED数据集评估指南

本指南将帮助你使用下载的NeuroLM模型文件来验证SEED数据集的效果。

## 文件结构要求

确保你的项目目录结构如下：

```
NeuroLM/
├── model/                    # 模型相关文件
│   ├── model_neurolm.py
│   ├── model.py
│   └── ...
├── downstream_dataset.py     # SEED数据集类
├── utils.py                  # 工具函数
├── evaluate_seed.py          # 评估脚本
└── your_model.pt            # 你下载的模型文件
```

## 数据集准备

### 1. SEED数据集格式

SEED数据集应该是一个HDF5文件，路径为：`your_dataset_dir/h5data/seed-3.hdf5`

数据集结构应该包含：
- 每个被试的EEG数据
- 情感标签（Positive, Neutral, Negative）
- 通道信息和采样频率

### 2. 数据集预处理

如果你还没有处理好的SEED数据集，可以参考 `dataset_maker/prepare_SEED.py` 来预处理原始数据。

## 使用方法

### 方法1：使用提供的评估脚本

```bash
python evaluate_seed.py \
    --model_path /path/to/your/model.pt \
    --dataset_dir /path/to/your/dataset \
    --device cuda \
    --batch_size 32
```

参数说明：
- `--model_path`: 你下载的模型文件路径（.pt文件）
- `--dataset_dir`: SEED数据集根目录
- `--device`: 使用的设备（cuda/cpu）
- `--batch_size`: 批次大小（根据GPU内存调整）

### 方法2：使用原始训练脚本

如果你有完整的训练环境，也可以使用原始的 `train_instruction.py` 脚本：

```bash
python train_instruction.py \
    --eval_only \
    --out_dir ./output \
    --dataset_dir /path/to/your/dataset \
    --NeuroLM_path /path/to/your/model.pt \
    --eeg_batch_size 32
```

## 模型文件说明

### 模型文件格式

下载的 `.pt` 文件是一个PyTorch检查点，包含：
- `model`: 模型权重
- `model_args`: 模型配置参数
- `optimizer`: 优化器状态（可选）

### 模型配置

NeuroLM模型的主要配置参数：
- `n_layer`: 12 (Transformer层数)
- `n_head`: 12 (注意力头数)
- `n_embd`: 768 (嵌入维度)
- `block_size`: 1024 (序列长度)
- `vocab_size`: 50257 (词汇表大小)

## 评估指标

SEED数据集评估将输出以下指标：
- **accuracy**: 准确率
- **balanced_accuracy**: 平衡准确率
- **cohen_kappa**: Cohen's Kappa系数
- **f1_weighted**: 加权F1分数

## 预期结果

根据论文报告，NeuroLM在SEED数据集上的典型性能：
- 准确率: ~85-90%
- 平衡准确率: ~85-90%
- F1分数: ~85-90%

## 常见问题

### 1. 内存不足
如果遇到GPU内存不足，可以：
- 减小 `batch_size`
- 使用CPU进行评估（设置 `--device cpu`）

### 2. 模型加载失败
确保：
- 模型文件路径正确
- 模型文件完整且未损坏
- 使用了正确的模型架构

### 3. 数据集加载失败
检查：
- 数据集路径是否正确
- HDF5文件是否存在且格式正确
- 是否有足够的磁盘空间

### 4. 依赖包缺失
安装必要的依赖：
```bash
pip install torch numpy tiktoken h5py einops
```

## 示例输出

成功运行后，你应该看到类似以下的输出：

```
使用设备: cuda
Loading model from /path/to/your/model.pt
Loading SEED dataset from /path/to/your/dataset
开始评估...
处理批次 0/50
处理批次 10/50
...
处理批次 40/50

==================================================
SEED数据集评估结果:
==================================================
accuracy: 0.8756
balanced_accuracy: 0.8723
cohen_kappa: 0.8134
f1_weighted: 0.8751
==================================================
```

## 进一步分析

如果你想要更详细的分析，可以：
1. 查看每个类别的性能
2. 分析混淆矩阵
3. 可视化预测结果
4. 比较不同模型配置的效果

## 技术支持

如果遇到问题，请检查：
1. 所有依赖包是否正确安装
2. 文件路径是否正确
3. 模型文件是否完整
4. 数据集格式是否符合要求 