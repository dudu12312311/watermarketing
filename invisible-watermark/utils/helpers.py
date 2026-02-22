"""
辅助函数 - 通用工具函数
"""

import os
import torch
import random
import numpy as np
from pathlib import Path


def set_seed(seed=42):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 确保卷积算法确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_name='cuda'):
    """
    获取计算设备
    
    Args:
        device_name: 设备名称 ('cuda' 或 'cpu')
    
    Returns:
        device: torch设备对象
    """
    if device_name == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮数
        loss: 当前损失
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path, device='cuda'):
    """
    加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        device: 设备
    
    Returns:
        epoch: 加载的轮数
        loss: 加载的损失
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch}, loss {loss:.6f}")
    
    return epoch, loss


def load_model_weights(model, weights_path, device='cuda'):
    """
    加载模型权重（不包括优化器状态）
    
    Args:
        model: 模型
        weights_path: 权重文件路径
        device: 设备
    """
    weights = torch.load(weights_path, map_location=device)
    
    if 'model_state_dict' in weights:
        model.load_state_dict(weights['model_state_dict'])
    else:
        model.load_state_dict(weights)
    
    print(f"Model weights loaded from {weights_path}")


def count_parameters(model):
    """
    计算模型参数数量
    
    Args:
        model: 模型
    
    Returns:
        total: 总参数数
        trainable: 可训练参数数
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


def print_model_info(model, model_name='Model'):
    """
    打印模型信息
    
    Args:
        model: 模型
        model_name: 模型名称
    """
    total, trainable = count_parameters(model)
    
    print(f"\n{'='*50}")
    print(f"{model_name} Information")
    print(f"{'='*50}")
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"{'='*50}\n")


def create_optimizer(model, optimizer_name='adam', learning_rate=1e-4, weight_decay=1e-5):
    """
    创建优化器
    
    Args:
        model: 模型
        optimizer_name: 优化器名称 ('adam', 'sgd', 'rmsprop')
        learning_rate: 学习率
        weight_decay: 权重衰减
    
    Returns:
        optimizer: 优化器
    """
    if optimizer_name.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    elif optimizer_name.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    return optimizer


def create_scheduler(optimizer, scheduler_name='cosine', num_epochs=300, warmup_epochs=10):
    """
    创建学习率调度器
    
    Args:
        optimizer: 优化器
        scheduler_name: 调度器名称 ('cosine', 'step', 'exponential')
        num_epochs: 总轮数
        warmup_epochs: 预热轮数
    
    Returns:
        scheduler: 学习率调度器
    """
    if scheduler_name.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs - warmup_epochs,
        )
    elif scheduler_name.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=num_epochs // 3,
            gamma=0.1,
        )
    elif scheduler_name.lower() == 'exponential':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=0.95,
        )
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    
    return scheduler


def adjust_learning_rate(optimizer, epoch, initial_lr, warmup_epochs=10):
    """
    调整学习率（预热）
    
    Args:
        optimizer: 优化器
        epoch: 当前轮数
        initial_lr: 初始学习率
        warmup_epochs: 预热轮数
    """
    if epoch < warmup_epochs:
        lr = initial_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def get_lr(optimizer):
    """
    获取当前学习率
    
    Args:
        optimizer: 优化器
    
    Returns:
        lr: 当前学习率
    """
    return optimizer.param_groups[0]['lr']


def format_time(seconds):
    """
    格式化时间
    
    Args:
        seconds: 秒数
    
    Returns:
        formatted: 格式化的时间字符串
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


if __name__ == '__main__':
    # 测试辅助函数
    print("Testing helper functions...")
    
    # 测试设置随机种子
    set_seed(42)
    print("Random seed set to 42")
    
    # 测试获取设备
    device = get_device('cuda')
    print(f"Device: {device}")
    
    # 测试模型参数计数
    from models.encoder import EncoderNet
    model = EncoderNet(message_length=32, hidden_channels=64, num_layers=4)
    print_model_info(model, "EncoderNet")
    
    # 测试优化器创建
    optimizer = create_optimizer(model, 'adam', 1e-4, 1e-5)
    print(f"Optimizer: {optimizer}")
    
    # 测试调度器创建
    scheduler = create_scheduler(optimizer, 'cosine', 300, 10)
    print(f"Scheduler: {scheduler}")
    
    # 测试时间格式化
    print(f"Time: {format_time(3661)}")
