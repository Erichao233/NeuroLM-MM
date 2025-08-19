"""
SEED数据集评估脚本
使用预训练的NeuroLM模型验证SEED数据集效果
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
import tiktoken

from model.model_neurolm import NeuroLM
from model.model import GPTConfig
from downstream_dataset import SEEDDataset
from utils import get_metrics


def init_model(model_path, tokenizer_path=None, device='cuda'):
    """初始化模型"""
    print(f"Loading model from {model_path}")
    
    # 加载模型检查点
    checkpoint = torch.load(model_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    # 模型配置
    model_args = dict(
        n_layer=12,
        n_head=12, 
        n_embd=768,
        block_size=1024,
        bias=False,
        vocab_size=50257,
        dropout=0.0
    )
    
    # 强制使用检查点中的配置
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    
    # 创建模型
    gptconf = GPTConfig(**model_args)
    model = NeuroLM(gptconf, tokenizer_path, init_from='scratch')
    
    # 加载模型权重
    state_dict = checkpoint['model']
    # 修复状态字典的键
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def get_seed_dataset(dataset_dir, eeg_max_len=276, text_max_len=80):
    """准备SEED数据集"""
    print(f"Loading SEED dataset from {dataset_dir}")
    
    # 创建数据集
    dataset_test = SEEDDataset(
        Path(dataset_dir, 'h5data/seed-3.hdf5'), 
        window_size=800, 
        stride_size=800, 
        trial_start_percentage=0.8, 
        trial_end_percentage=1, 
        is_instruct=True, 
        is_val=True, 
        eeg_max_len=eeg_max_len, 
        text_max_len=text_max_len
    )
    
    # 创建数据加载器
    data_loader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=32,  # 可以根据GPU内存调整
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        shuffle=False
    )
    
    # 数据集信息
    dataset_info = {
        'name': 'SEED',
        'metrics': ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"],
        'is_binary': False,
        'num_classes': 3,
        'result_idx': 11,
        'label_dic': {'Positive': 0, 'Neutral': 1, 'Negative': 2}
    }
    
    return data_loader, dataset_info


def get_pred(pred_string, dataset_info):
    """从预测字符串中提取预测结果"""
    pred = -1
    try:
        pred = pred_string.split(' ')[dataset_info['result_idx']]
        if pred.startswith('('):
            pred = pred[:3]
        pred = dataset_info['label_dic'][pred]
    except:
        print(f'无法解析预测结果: {pred_string}')
        pred = -1
    return pred


@torch.no_grad()
def evaluate(model, dataset_info, dataloader, device):
    """评估模型"""
    print("开始评估...")
    
    model.eval()
    preds = []
    targets = []
    
    # 初始化tokenizer
    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx % 10 == 0:
            print(f"处理批次 {batch_idx}/{len(dataloader)}")
        
        X_eeg, X_text, label, input_chans, input_time, input_mask, gpt_mask = batch
        
        # 移动到设备
        X_eeg = X_eeg.float().to(device, non_blocking=True)
        X_text = X_text.to(device, non_blocking=True)
        input_chans = input_chans.to(device, non_blocking=True)
        input_time = input_time.to(device, non_blocking=True)
        gpt_mask = gpt_mask.to(device, non_blocking=True)
        if input_mask is not None:
            input_mask = input_mask.to(device, non_blocking=True)
        
        # 生成预测
        text = model.generate(
            X_eeg, X_text, input_chans, input_time, input_mask, 
            eeg_text_mask=gpt_mask, max_new_tokens=5
        )
        text = text[:, 1:]  # 移除[SEP] token
        
        # 解析预测结果
        for i, t in enumerate(text):
            pred_string = decode(t.tolist())
            pred = get_pred(pred_string, dataset_info)
            
            if not dataset_info['is_binary']:
                pred = np.eye(dataset_info['num_classes'])[pred]
            preds.append(pred)
        
        targets.append(label)
    
    # 计算指标
    targets = torch.cat(targets, dim=0).numpy()
    preds = np.array(preds)
    
    results = get_metrics(preds, targets, dataset_info['metrics'], dataset_info['is_binary'])
    
    return results


def main():
    parser = argparse.ArgumentParser(description='SEED数据集评估')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径(.pt)')
    parser.add_argument('--tokenizer_path', type=str, default=None, help='tokenizer文件路径')
    parser.add_argument('--dataset_dir', type=str, required=True, help='SEED数据集目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.dataset_dir):
        print(f"错误: 数据集目录不存在: {args.dataset_dir}")
        return
    
    # 设置设备
    device = args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    print(f"使用设备: {device}")
    
    try:
        # 初始化模型
        model = init_model(args.model_path, args.tokenizer_path, device)
        
        # 准备数据集
        dataloader, dataset_info = get_seed_dataset(args.dataset_dir)
        
        # 评估模型
        results = evaluate(model, dataset_info, dataloader, device)
        
        # 打印结果
        print("\n" + "="*50)
        print("SEED数据集评估结果:")
        print("="*50)
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
        print("="*50)
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main() 