"""
测试脚本 - 测试隐形水印系统的性能
"""

import os
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

from config import get_config
from models import EncoderNet, DecoderNet, NoiseLayerContainer
from utils import (
    MetricsRecorder, calculate_psnr, calculate_ssim, calculate_bitwise_error,
    get_device, load_model_weights
)
from data import WatermarkDataset
from torch.utils.data import DataLoader


class Tester:
    """测试器类"""
    
    def __init__(self, encoder_path, decoder_path, device='cuda'):
        """
        初始化测试器
        
        Args:
            encoder_path: 编码器模型路径
            decoder_path: 解码器模型路径
            device: 计算设备
        """
        self.device = device
        config = get_config()
        
        # 加载编码器
        print("Loading encoder...")
        self.encoder = EncoderNet(
            message_length=config['model']['message_length'],
            hidden_channels=config['model']['encoder']['hidden_channels'],
            num_layers=config['model']['encoder']['num_layers'],
        ).to(device)
        load_model_weights(self.encoder, encoder_path, device)
        self.encoder.eval()
        
        # 加载解码器
        print("Loading decoder...")
        self.decoder = DecoderNet(
            message_length=config['model']['message_length'],
            hidden_channels=config['model']['decoder']['hidden_channels'],
            num_layers=config['model']['decoder']['num_layers'],
        ).to(device)
        load_model_weights(self.decoder, decoder_path, device)
        self.decoder.eval()
        
        self.config = config
    
    def test_no_attack(self, test_loader):
        """
        测试无攻击情况
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            metrics: 指标字典
        """
        print("\n" + "="*60)
        print("Testing without attack")
        print("="*60)
        
        metrics = MetricsRecorder()
        
        with torch.no_grad():
            for images, messages in tqdm(test_loader):
                images = images.to(self.device)
                messages = messages.to(self.device)
                
                # 编码
                watermarked = self.encoder(images, messages)
                
                # 解码
                decoded = self.decoder(watermarked)
                
                # 计算指标
                psnr = calculate_psnr(watermarked, images)
                ssim = calculate_ssim(watermarked, images)
                ber = calculate_bitwise_error(decoded, messages)
                mse = torch.mean((watermarked - images) ** 2).item()
                
                metrics.update(psnr=psnr, ssim=ssim, ber=ber, mse=mse)
        
        return metrics.get_all_averages()
    
    def test_with_noise(self, test_loader, noise_config):
        """
        测试有噪声情况
        
        Args:
            test_loader: 测试数据加载器
            noise_config: 噪声配置
        
        Returns:
            metrics: 指标字典
        """
        print(f"\n" + "="*60)
        print(f"Testing with noise: {noise_config}")
        print("="*60)
        
        # 创建噪声层
        noise_layer = NoiseLayerContainer(noise_config, use_random=False).to(self.device)
        
        metrics = MetricsRecorder()
        
        with torch.no_grad():
            for images, messages in tqdm(test_loader):
                images = images.to(self.device)
                messages = messages.to(self.device)
                
                # 编码
                watermarked = self.encoder(images, messages)
                
                # 应用噪声
                attacked = noise_layer(watermarked)
                
                # 解码
                decoded = self.decoder(attacked)
                
                # 计算指标
                psnr = calculate_psnr(attacked, images)
                ssim = calculate_ssim(attacked, images)
                ber = calculate_bitwise_error(decoded, messages)
                mse = torch.mean((attacked - images) ** 2).item()
                
                metrics.update(psnr=psnr, ssim=ssim, ber=ber, mse=mse)
        
        return metrics.get_all_averages()
    
    def test_all(self, test_loader):
        """
        测试所有场景
        
        Args:
            test_loader: 测试数据加载器
        
        Returns:
            results: 测试结果字典
        """
        results = {}
        
        # 无攻击
        results['no_attack'] = self.test_no_attack(test_loader)
        
        # 单个噪声层
        noise_configs = [
            'crop((0.2,0.25),(0.2,0.25))',
            'cropout((0.55,0.6),(0.55,0.6))',
            'dropout(0.55,0.6)',
            'resize(0.7,0.8)',
            'jpeg()',
        ]
        
        results['single_noise'] = {}
        for config in noise_configs:
            results['single_noise'][config] = self.test_with_noise(test_loader, config)
        
        # 组合噪声
        combined_config = 'crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()'
        results['combined_noise'] = self.test_with_noise(test_loader, combined_config)
        
        return results
    
    def print_results(self, results):
        """
        打印测试结果
        
        Args:
            results: 测试结果字典
        """
        print("\n" + "="*80)
        print("Test Results Summary")
        print("="*80)
        
        # 无攻击
        print("\nNo Attack:")
        print("-" * 80)
        print(f"{'Metric':<20} {'Value':<20}")
        print("-" * 80)
        for key, value in results['no_attack'].items():
            print(f"{key:<20} {value:<20.6f}")
        
        # 单个噪声层
        print("\nSingle Noise Layers:")
        print("-" * 80)
        print(f"{'Noise Type':<30} {'PSNR':<15} {'SSIM':<15} {'BER':<15}")
        print("-" * 80)
        for noise_type, metrics in results['single_noise'].items():
            print(f"{noise_type:<30} {metrics['psnr']:<15.2f} {metrics['ssim']:<15.4f} {metrics['ber']:<15.6f}")
        
        # 组合噪声
        print("\nCombined Noise:")
        print("-" * 80)
        print(f"{'Metric':<20} {'Value':<20}")
        print("-" * 80)
        for key, value in results['combined_noise'].items():
            print(f"{key:<20} {value:<20.6f}")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Test HiDDeN watermarking system')
    parser.add_argument('--encoder', type=str, required=True, help='Encoder model path')
    parser.add_argument('--decoder', type=str, required=True, help='Decoder model path')
    parser.add_argument('--test-dir', type=str, default='data/val', help='Test data directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--output', type=str, help='Output file for results')
    
    args = parser.parse_args()
    
    # 获取设备
    device = get_device(args.device)
    
    # 获取配置
    config = get_config()
    
    # 创建测试数据加载器
    print("Loading test data...")
    try:
        test_dataset = WatermarkDataset(
            args.test_dir,
            image_size=config['data']['image_size'],
            message_length=config['model']['message_length'],
            random_crop=False,
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        
        print(f"Test samples: {len(test_dataset)}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # 创建测试器
    tester = Tester(args.encoder, args.decoder, device)
    
    # 运行测试
    results = tester.test_all(test_loader)
    
    # 打印结果
    tester.print_results(results)
    
    # 保存结果
    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
