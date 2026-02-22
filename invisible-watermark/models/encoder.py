"""
编码器网络 - 将消息隐藏到图像中
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderNet(nn.Module):
    """
    编码器网络：将二进制消息隐藏到图像中
    
    输入：
        - image: (B, 3, H, W) - 原始图像
        - message: (B, message_length) - 二进制消息
    
    输出：
        - watermarked_image: (B, 3, H, W) - 包含隐藏消息的水印图像
    """
    
    def __init__(self, message_length=32, hidden_channels=64, num_layers=4):
        """
        初始化编码器
        
        Args:
            message_length: 隐藏消息的长度（比特数）
            hidden_channels: 隐藏层的通道数
            num_layers: 编码器的层数
        """
        super(EncoderNet, self).__init__()
        
        self.message_length = message_length
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # 消息编码层 - 将消息扩展到空间维度
        # 消息 (B, message_length) -> (B, hidden_channels, H, W)
        self.message_encoder = nn.Sequential(
            nn.Linear(message_length, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, hidden_channels * 4 * 4),
            nn.ReLU(inplace=True),
        )
        
        # 主编码器网络
        layers = []
        
        # 第一层：3 -> hidden_channels
        layers.append(nn.Conv2d(3 + hidden_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(hidden_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层
        for i in range(num_layers - 2):
            layers.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(hidden_channels))
            layers.append(nn.ReLU(inplace=True))
        
        # 最后一层：hidden_channels -> 3
        layers.append(nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1))
        
        self.encoder = nn.Sequential(*layers)
        
        # 残差连接的缩放因子
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, image, message):
        """
        前向传播
        
        Args:
            image: (B, 3, H, W) - 原始图像
            message: (B, message_length) - 二进制消息
        
        Returns:
            watermarked_image: (B, 3, H, W) - 水印图像
        """
        batch_size, _, height, width = image.shape
        
        # 编码消息
        encoded_message = self.message_encoder(message)  # (B, hidden_channels * 16)
        encoded_message = encoded_message.view(batch_size, self.hidden_channels, 4, 4)
        
        # 上采样消息到图像大小
        encoded_message = F.interpolate(
            encoded_message,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        
        # 连接图像和编码的消息
        combined = torch.cat([image, encoded_message], dim=1)
        
        # 通过编码器
        residual = self.encoder(combined)
        
        # 添加残差连接
        watermarked_image = image + self.residual_scale * residual
        
        # 限制输出范围到 [-1, 1]
        watermarked_image = torch.tanh(watermarked_image)
        
        return watermarked_image


class EncoderNetV2(nn.Module):
    """
    改进的编码器网络 - 使用更深的网络和更多的残差连接
    """
    
    def __init__(self, message_length=32, hidden_channels=64, num_layers=6):
        """
        初始化改进的编码器
        
        Args:
            message_length: 隐藏消息的长度（比特数）
            hidden_channels: 隐藏层的通道数
            num_layers: 编码器的层数
        """
        super(EncoderNetV2, self).__init__()
        
        self.message_length = message_length
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        
        # 消息编码层
        self.message_encoder = nn.Sequential(
            nn.Linear(message_length, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, hidden_channels * 8 * 8),
            nn.ReLU(inplace=True),
        )
        
        # 主编码器网络 - 使用残差块
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3 + hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        
        # 残差块
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_channels) for _ in range(num_layers - 2)
        ])
        
        # 最后一层
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        
        self.residual_scale = nn.Parameter(torch.tensor(0.1))
    
    def _make_residual_block(self, channels):
        """创建残差块"""
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
        )
    
    def forward(self, image, message):
        """
        前向传播
        
        Args:
            image: (B, 3, H, W) - 原始图像
            message: (B, message_length) - 二进制消息
        
        Returns:
            watermarked_image: (B, 3, H, W) - 水印图像
        """
        batch_size, _, height, width = image.shape
        
        # 编码消息
        encoded_message = self.message_encoder(message)
        encoded_message = encoded_message.view(batch_size, self.hidden_channels, 8, 8)
        
        # 上采样消息到图像大小
        encoded_message = F.interpolate(
            encoded_message,
            size=(height, width),
            mode='bilinear',
            align_corners=False
        )
        
        # 连接图像和编码的消息
        combined = torch.cat([image, encoded_message], dim=1)
        
        # 初始卷积
        x = self.initial_conv(combined)
        
        # 残差块
        for block in self.residual_blocks:
            residual = x
            x = block(x)
            x = x + residual
            x = F.relu(x)
        
        # 最后一层
        residual = self.final_conv(x)
        
        # 添加残差连接
        watermarked_image = image + self.residual_scale * residual
        
        return watermarked_image


if __name__ == '__main__':
    # 测试编码器
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    encoder = EncoderNet(message_length=32, hidden_channels=64, num_layers=4).to(device)
    
    # 创建测试输入
    batch_size = 2
    image = torch.randn(batch_size, 3, 400, 400).to(device)
    message = torch.randint(0, 2, (batch_size, 32)).float().to(device)
    
    # 前向传播
    watermarked = encoder(image, message)
    
    print(f"Input image shape: {image.shape}")
    print(f"Message shape: {message.shape}")
    print(f"Watermarked image shape: {watermarked.shape}")
    print(f"Watermarked image range: [{watermarked.min():.4f}, {watermarked.max():.4f}]")
    
    # 计算MSE
    mse = torch.mean((watermarked - image) ** 2)
    print(f"Encoder MSE: {mse.item():.6f}")
