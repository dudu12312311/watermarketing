"""
解码器网络 - 从水印图像中恢复隐藏的消息
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderNet(nn.Module):
    """
    解码器网络：从水印图像中恢复隐藏的消息
    
    输入：
        - watermarked_image: (B, 3, H, W) - 可能被攻击的水印图像
    
    输出：
        - message: (B, message_length) - 恢复的二进制消息
    """
    
    def __init__(self, message_length=32, hidden_channels=64, num_layers=4):
        """
        初始化解码器
        
        Args:
            message_length: 隐藏消息的长度（比特数）
            hidden_channels: 隐藏层的通道数
            num_layers: 解码器的层数
        """
        super(DecoderNet, self).__init__()
        
        self.message_length = message_length
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # 特征提取网络
        layers = []
        
        # 第一层：3 -> hidden_channels
        layers.append(nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层
        for i in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # 最后一层卷积
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 消息解码层
        self.message_decoder = nn.Sequential(
            nn.Linear(hidden_channels, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, message_length),
            nn.Sigmoid(),  # 输出范围 [0, 1]
        )
    
    def forward(self, watermarked_image):
        """
        前向传播
        
        Args:
            watermarked_image: (B, 3, H, W) - 水印图像
        
        Returns:
            message: (B, message_length) - 恢复的消息
        """
        # 特征提取
        features = self.feature_extractor(watermarked_image)  # (B, hidden_channels, H, W)
        
        # 全局平均池化
        pooled = self.global_avg_pool(features)  # (B, hidden_channels, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, hidden_channels)
        
        # 消息解码
        message = self.message_decoder(pooled)  # (B, message_length)
        
        return message


class DecoderNetV2(nn.Module):
    """
    改进的解码器网络 - 使用更深的网络和多尺度特征
    """
    
    def __init__(self, message_length=32, hidden_channels=64, num_layers=6):
        """
        初始化改进的解码器
        
        Args:
            message_length: 隐藏消息的长度（比特数）
            hidden_channels: 隐藏层的通道数
            num_layers: 解码器的层数
        """
        super(DecoderNetV2, self).__init__()
        
        self.message_length = message_length
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # 初始卷积
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_channels) for _ in range(num_layers - 2)
        ])
        
        # 最后一层卷积
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 消息解码层
        self.message_decoder = nn.Sequential(
            nn.Linear(hidden_channels, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, message_length),
            nn.Sigmoid(),
        )
    
    def _make_residual_block(self, channels):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, watermarked_image):
        """
        前向传播
        
        Args:
            watermarked_image: (B, 3, H, W) - 水印图像
        
        Returns:
            message: (B, message_length) - 恢复的消息
        """
        # 初始卷积
        x = self.initial_conv(watermarked_image)
        
        # 残差块
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = x + residual
            x = F.relu(x)
        
        # 最后一层卷积
        x = self.final_conv(x)
        
        # 全局平均池化
        pooled = self.global_avg_pool(x)
        pooled = pooled.view(pooled.size(0), -1)
        
        # 消息解码
        message = self.message_decoder(pooled)
        
        return message


if __name__ == '__main__':
    # 测试解码器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    decoder = DecoderNet(message_length=32, hidden_channels=64, num_layers=4).to(device)
    
    # 创建测试输入
    batch_size = 2
    watermarked_image = torch.randn(batch_size, 3, 400, 400).to(device)
    
    # 前向传播
    message = decoder(watermarked_image)
    
    print(f"Watermarked image shape: {watermarked_image.shape}")
    print(f"Recovered message shape: {message.shape}")
    print(f"Message range: [{message.min():.4f}, {message.max():.4f}]")
