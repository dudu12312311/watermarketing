"""
HiDDeN隐形水印系统配置文件
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# ==================== 数据配置 ====================
DATA_CONFIG = {
    'train_dir': 'data/train',
    'val_dir': 'data/val',
    'image_size': 400,  # 图像大小
    'batch_size': 32,
    'num_workers': 4,
    'pin_memory': False,  # 如果有GPU可以设为True
}

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    'encoder': {
        'input_channels': 3,
        'output_channels': 3,
        'hidden_channels': 64,
        'num_layers': 4,
    },
    'decoder': {
        'input_channels': 3,
        'hidden_channels': 64,
        'num_layers': 4,
    },
    'message_length': 32,  # 隐藏消息长度（比特）
}

# ==================== 训练配置 ====================
TRAIN_CONFIG = {
    'num_epochs': 300,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'optimizer': 'adam',
    'scheduler': 'cosine',  # 'cosine' 或 'step'
    'warmup_epochs': 10,
    'log_interval': 100,
    'save_interval': 10,  # 每N个epoch保存一次
    'device': 'cuda',  # 'cuda' 或 'cpu'
}

# ==================== 损失函数配置 ====================
LOSS_CONFIG = {
    'encoder_mse_weight': 1.0,
    'decoder_bce_weight': 1.0,
    'use_l1_loss': False,
}

# ==================== 噪声层配置 ====================
NOISE_CONFIG = {
    'noise_layers': [
        'crop((0.2,0.25),(0.2,0.25))',
        'cropout((0.55,0.6),(0.55,0.6))',
        'dropout(0.55,0.6)',
        'resize(0.7,0.8)',
        'jpeg()',
    ],
    'combined_noise': 'crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()',
    'use_random_noise': True,  # 随机选择噪声层而非顺序应用
}

# ==================== 评估配置 ====================
EVAL_CONFIG = {
    'eval_interval': 1,  # 每N个epoch评估一次
    'save_best_model': True,
    'early_stopping_patience': 50,
}

# ==================== 日志配置 ====================
LOG_CONFIG = {
    'log_dir': 'logs',
    'log_interval': 10,  # 每N个batch记录一次
    'tensorboard': True,
    'save_images': True,
    'num_save_images': 8,
}

# ==================== 推理配置 ====================
INFERENCE_CONFIG = {
    'device': 'cuda',
    'batch_size': 1,
    'num_workers': 0,
}

# ==================== 路径配置 ====================
PATHS = {
    'models': PROJECT_ROOT / 'checkpoints',
    'logs': PROJECT_ROOT / 'logs',
    'results': PROJECT_ROOT / 'results',
    'data': PROJECT_ROOT / 'data',
}

# 创建必要的目录
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)


def get_config():
    """获取完整配置"""
    return {
        'data': DATA_CONFIG,
        'model': MODEL_CONFIG,
        'train': TRAIN_CONFIG,
        'loss': LOSS_CONFIG,
        'noise': NOISE_CONFIG,
        'eval': EVAL_CONFIG,
        'log': LOG_CONFIG,
        'inference': INFERENCE_CONFIG,
        'paths': PATHS,
    }


if __name__ == '__main__':
    config = get_config()
    import json
    print(json.dumps({k: str(v) if isinstance(v, Path) else v for k, v in config.items()}, indent=2))
