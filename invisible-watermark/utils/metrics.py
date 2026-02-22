"""
评估指标 - 计算模型性能指标
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


def calculate_psnr(img1, img2, max_val=1.0):
    """
    计算峰值信噪比 (PSNR)
    
    Args:
        img1: 图像1 (B, C, H, W) 或 (C, H, W)
        img2: 图像2 (B, C, H, W) 或 (C, H, W)
        max_val: 像素最大值
    
    Returns:
        psnr: PSNR值（dB）
    """
    mse = torch.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    
    return psnr.item()


def calculate_ssim(img1, img2, window_size=11, sigma=1.5, max_val=1.0):
    """
    计算结构相似性指数 (SSIM)
    
    Args:
        img1: 图像1 (B, C, H, W) 或 (C, H, W)
        img2: 图像2 (B, C, H, W) 或 (C, H, W)
        window_size: 高斯窗口大小
        sigma: 高斯标准差
        max_val: 像素最大值
    
    Returns:
        ssim: SSIM值 (0-1)
    """
    # 确保输入是4D张量
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    batch_size, channels, height, width = img1.shape
    
    # 创建高斯窗口
    kernel = _create_gaussian_kernel(window_size, sigma, channels)
    kernel = kernel.to(img1.device)
    
    # 计算均值
    mu1 = F.conv2d(img1, kernel, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(img2, kernel, padding=window_size//2, groups=channels)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(img1 ** 2, kernel, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, kernel, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=window_size//2, groups=channels) - mu1_mu2
    
    # 计算SSIM
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    ssim = ssim_map.mean()
    
    return ssim.item()


def calculate_bitwise_error(predicted, target, threshold=0.5):
    """
    计算比特级错误率
    
    Args:
        predicted: 预测的消息 (B, message_length) 或 (message_length,)
        target: 目标消息 (B, message_length) 或 (message_length,)
        threshold: 二值化阈值
    
    Returns:
        error_rate: 错误率 (0-1)
    """
    # 二值化
    predicted_binary = (predicted > threshold).float()
    target_binary = (target > 0.5).float()
    
    # 计算错误
    errors = torch.abs(predicted_binary - target_binary)
    error_rate = errors.mean()
    
    return error_rate.item()


def _create_gaussian_kernel(window_size, sigma, channels):
    """创建高斯核"""
    x = torch.arange(window_size).float() - (window_size - 1) / 2
    gauss = torch.exp(-x.pow(2.0) / (2 * sigma ** 2))
    kernel = gauss / gauss.sum()
    
    kernel = kernel.unsqueeze(1) * kernel.unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(channels, 1, 1, 1)
    
    return kernel


class MetricsRecorder:
    """
    指标记录器 - 记录和计算平均指标
    """
    
    def __init__(self):
        self.metrics = defaultdict(list)
    
    def update(self, **kwargs):
        """
        更新指标
        
        Args:
            **kwargs: 指标名称和值的键值对
        """
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item() if value.numel() == 1 else value.mean().item()
            self.metrics[key].append(value)
    
    def get_average(self, key):
        """获取平均值"""
        if key not in self.metrics or len(self.metrics[key]) == 0:
            return 0.0
        return np.mean(self.metrics[key])
    
    def get_all_averages(self):
        """获取所有平均值"""
        return {key: self.get_average(key) for key in self.metrics.keys()}
    
    def reset(self):
        """重置所有指标"""
        self.metrics.clear()
    
    def __str__(self):
        """字符串表示"""
        averages = self.get_all_averages()
        lines = []
        for key, value in averages.items():
            lines.append(f"{key}: {value:.6f}")
        return '\n'.join(lines)


if __name__ == '__main__':
    # 测试指标计算
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    img1 = torch.randn(2, 3, 100, 100).to(device)
    img2 = img1 + torch.randn_like(img1) * 0.01
    
    # 计算PSNR
    psnr = calculate_psnr(img1, img2)
    print(f"PSNR: {psnr:.2f} dB")
    
    # 计算SSIM
    ssim = calculate_ssim(img1, img2)
    print(f"SSIM: {ssim:.4f}")
    
    # 计算比特级错误率
    msg1 = torch.randint(0, 2, (2, 32)).float().to(device)
    msg2 = msg1.clone()
    msg2[0, 0] = 1 - msg2[0, 0]  # 翻转一个比特
    
    ber = calculate_bitwise_error(msg1, msg2)
    print(f"Bitwise Error Rate: {ber:.4f}")
    
    # 测试指标记录器
    recorder = MetricsRecorder()
    for i in range(10):
        recorder.update(
            loss=torch.tensor(0.5 - i * 0.01),
            psnr=torch.tensor(30 + i),
            ssim=torch.tensor(0.8 + i * 0.01),
        )
    
    print("\nMetrics Recorder:")
    print(recorder)
