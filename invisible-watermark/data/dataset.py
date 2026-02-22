"""
数据加载器 - 为隐形水印系统加载和预处理数据
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class WatermarkDataset(Dataset):
    """
    隐形水印数据集
    
    从指定目录加载图像，并生成随机二进制消息
    """
    
    def __init__(self, image_dir, image_size=400, message_length=32, 
                 random_crop=True, transform=None):
        """
        初始化数据集
        
        Args:
            image_dir: 图像目录路径
            image_size: 图像大小（正方形）
            message_length: 消息长度（比特数）
            random_crop: 是否使用随机裁剪
            transform: 额外的图像变换
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.message_length = message_length
        self.random_crop = random_crop
        
        # 获取所有图像文件
        self.image_files = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    self.image_files.append(os.path.join(root, file))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {image_dir}")
        
        # 基础变换
        self.base_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # 额外变换
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        """
        获取数据项
        
        Returns:
            image: (3, H, W) - 图像张量
            message: (message_length,) - 二进制消息
        """
        # 加载图像
        image_path = self.image_files[idx]
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # 返回随机图像作为备选
            image = Image.new('RGB', (self.image_size, self.image_size))
        
        # 调整大小
        if self.random_crop:
            # 随机裁剪
            image = self._random_crop(image)
        else:
            # 中心裁剪
            image = self._center_crop(image)
        
        # 应用变换
        image = self.base_transform(image)
        
        if self.transform:
            image = self.transform(image)
        
        # 生成随机消息
        message = torch.randint(0, 2, (self.message_length,)).float()
        
        return image, message
    
    def _random_crop(self, image):
        """随机裁剪图像"""
        width, height = image.size
        
        if width < self.image_size or height < self.image_size:
            # 如果图像太小，先放大
            scale = max(self.image_size / width, self.image_size / height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.BILINEAR)
            width, height = image.size
        
        # 随机裁剪
        left = random.randint(0, width - self.image_size)
        top = random.randint(0, height - self.image_size)
        image = image.crop((left, top, left + self.image_size, top + self.image_size))
        
        return image
    
    def _center_crop(self, image):
        """中心裁剪图像"""
        width, height = image.size
        
        if width < self.image_size or height < self.image_size:
            # 如果图像太小，先放大
            scale = max(self.image_size / width, self.image_size / height)
            new_size = (int(width * scale), int(height * scale))
            image = image.resize(new_size, Image.BILINEAR)
            width, height = image.size
        
        # 中心裁剪
        left = (width - self.image_size) // 2
        top = (height - self.image_size) // 2
        image = image.crop((left, top, left + self.image_size, top + self.image_size))
        
        return image


def create_dataloaders(train_dir, val_dir, batch_size=32, num_workers=4, 
                       image_size=400, message_length=32, pin_memory=True):
    """
    创建训练和验证数据加载器
    
    Args:
        train_dir: 训练图像目录
        val_dir: 验证图像目录
        batch_size: 批大小
        num_workers: 数据加载线程数
        image_size: 图像大小
        message_length: 消息长度
        pin_memory: 是否将数据固定在内存中
    
    Returns:
        train_loader, val_loader
    """
    # 训练数据集（使用随机裁剪）
    train_dataset = WatermarkDataset(
        train_dir,
        image_size=image_size,
        message_length=message_length,
        random_crop=True,
    )
    
    # 验证数据集（使用中心裁剪）
    val_dataset = WatermarkDataset(
        val_dir,
        image_size=image_size,
        message_length=message_length,
        random_crop=False,
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # 测试数据加载器
    import sys
    
    # 创建测试数据目录
    test_dir = 'test_data'
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建一些测试图像
    print("Creating test images...")
    for i in range(5):
        img = Image.new('RGB', (500, 500), color=(i*50, i*50, i*50))
        img.save(os.path.join(test_dir, f'test_{i}.jpg'))
    
    # 创建数据集
    print("Creating dataset...")
    dataset = WatermarkDataset(test_dir, image_size=400, message_length=32)
    
    print(f"Dataset size: {len(dataset)}")
    
    # 获取一个样本
    image, message = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Message shape: {message.shape}")
    print(f"Message: {message}")
    
    # 创建数据加载器
    print("\nCreating dataloader...")
    loader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    for batch_idx, (images, messages) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print(f"  Images shape: {images.shape}")
        print(f"  Messages shape: {messages.shape}")
        if batch_idx == 0:
            break
    
    # 清理测试数据
    import shutil
    shutil.rmtree(test_dir)
    print("\nTest completed!")
