"""
训练脚本 - 训练HiDDeN隐形水印系统
"""

import os
import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import get_config
from models import EncoderNet, DecoderNet, NoiseLayerContainer
from data import WatermarkDataset
from torch.utils.data import DataLoader
from utils import (
    EncoderLoss, DecoderLoss, CombinedLoss,
    MetricsRecorder, calculate_psnr, calculate_ssim, calculate_bitwise_error,
    set_seed, get_device, save_checkpoint, load_checkpoint,
    count_parameters, print_model_info, create_optimizer, create_scheduler,
    adjust_learning_rate, get_lr, format_time
)


class Trainer:
    """训练器类"""
    
    def __init__(self, config, device, experiment_name=None):
        """
        初始化训练器
        
        Args:
            config: 配置字典
            device: 计算设备
            experiment_name: 实验名称
        """
        self.config = config
        self.device = device
        
        # 创建实验目录
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_dir = Path(config['paths']['logs']) / experiment_name
        self.checkpoint_dir = Path(config['paths']['models']) / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Experiment directory: {self.experiment_dir}")
        print(f"Checkpoint directory: {self.checkpoint_dir}")
        
        # 初始化模型
        self.encoder = EncoderNet(
            message_length=config['model']['message_length'],
            hidden_channels=config['model']['encoder']['hidden_channels'],
            num_layers=config['model']['encoder']['num_layers'],
        ).to(device)
        
        self.decoder = DecoderNet(
            message_length=config['model']['message_length'],
            hidden_channels=config['model']['decoder']['hidden_channels'],
            num_layers=config['model']['decoder']['num_layers'],
        ).to(device)
        
        # 打印模型信息
        print_model_info(self.encoder, "EncoderNet")
        print_model_info(self.decoder, "DecoderNet")
        
        # 初始化噪声层
        self.noise_layer = NoiseLayerContainer(
            config['noise']['noise_layers'],
            use_random=config['noise']['use_random_noise']
        ).to(device)
        
        # 初始化损失函数
        self.loss_fn = CombinedLoss(
            encoder_mse_weight=config['loss']['encoder_mse_weight'],
            encoder_bce_weight=config['loss']['decoder_bce_weight'],
            decoder_weight=1.0,
            use_l1=config['loss']['use_l1_loss']
        )
        
        # 初始化优化器
        self.encoder_optimizer = create_optimizer(
            self.encoder,
            config['train']['optimizer'],
            config['train']['learning_rate'],
            config['train']['weight_decay']
        )
        
        self.decoder_optimizer = create_optimizer(
            self.decoder,
            config['train']['optimizer'],
            config['train']['learning_rate'],
            config['train']['weight_decay']
        )
        
        # 初始化学习率调度器
        self.encoder_scheduler = create_scheduler(
            self.encoder_optimizer,
            config['train']['scheduler'],
            config['train']['num_epochs'],
            config['train']['warmup_epochs']
        )
        
        self.decoder_scheduler = create_scheduler(
            self.decoder_optimizer,
            config['train']['scheduler'],
            config['train']['num_epochs'],
            config['train']['warmup_epochs']
        )
        
        # 初始化TensorBoard
        if config['log']['tensorboard']:
            self.writer = SummaryWriter(str(self.experiment_dir))
        else:
            self.writer = None
        
        # 指标记录器
        self.train_metrics = MetricsRecorder()
        self.val_metrics = MetricsRecorder()
        
        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
    
    def train_epoch(self, train_loader, epoch):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch
        
        Returns:
            avg_loss: 平均损失
        """
        self.encoder.train()
        self.decoder.train()
        self.train_metrics.reset()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['train']['num_epochs']}")
        
        for batch_idx, (images, messages) in enumerate(pbar):
            images = images.to(self.device)
            messages = messages.to(self.device)
            
            # 调整学习率（预热）
            if epoch < self.config['train']['warmup_epochs']:
                adjust_learning_rate(
                    self.encoder_optimizer,
                    epoch,
                    self.config['train']['learning_rate'],
                    self.config['train']['warmup_epochs']
                )
                adjust_learning_rate(
                    self.decoder_optimizer,
                    epoch,
                    self.config['train']['learning_rate'],
                    self.config['train']['warmup_epochs']
                )
            
            # 前向传播
            watermarked_images = self.encoder(images, messages)
            
            # 应用噪声层
            attacked_images = self.noise_layer(watermarked_images)
            
            # 解码
            decoded_messages = self.decoder(watermarked_images)
            decoded_messages_attacked = self.decoder(attacked_images)
            
            # 计算损失
            loss, loss_dict = self.loss_fn(
                watermarked_images, images,
                decoded_messages, messages,
                decoded_messages_attacked
            )
            
            # 反向传播
            self.encoder_optimizer.zero_grad()
            self.decoder_optimizer.zero_grad()
            loss.backward()
            self.encoder_optimizer.step()
            self.decoder_optimizer.step()
            
            # 记录指标
            self.train_metrics.update(**loss_dict)
            
            # 计算额外指标
            with torch.no_grad():
                psnr = calculate_psnr(watermarked_images, images)
                ssim = calculate_ssim(watermarked_images, images)
                ber = calculate_bitwise_error(decoded_messages, messages)
                
                self.train_metrics.update(psnr=psnr, ssim=ssim, ber=ber)
            
            # 更新进度条
            pbar.set_postfix({
                'loss': loss.item(),
                'psnr': psnr,
                'ber': ber,
            })
            
            # 定期记录到TensorBoard
            if self.writer and batch_idx % self.config['log']['log_interval'] == 0:
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('train/loss', loss.item(), global_step)
                self.writer.add_scalar('train/psnr', psnr, global_step)
                self.writer.add_scalar('train/ssim', ssim, global_step)
                self.writer.add_scalar('train/ber', ber, global_step)
        
        # 返回平均损失
        avg_loss = self.train_metrics.get_average('total')
        return avg_loss
    
    def validate(self, val_loader, epoch):
        """
        验证
        
        Args:
            val_loader: 验证数据加载器
            epoch: 当前epoch
        
        Returns:
            avg_loss: 平均损失
        """
        self.encoder.eval()
        self.decoder.eval()
        self.val_metrics.reset()
        
        pbar = tqdm(val_loader, desc="Validating")
        
        with torch.no_grad():
            for images, messages in pbar:
                images = images.to(self.device)
                messages = messages.to(self.device)
                
                # 前向传播
                watermarked_images = self.encoder(images, messages)
                attacked_images = self.noise_layer(watermarked_images)
                
                # 解码
                decoded_messages = self.decoder(watermarked_images)
                decoded_messages_attacked = self.decoder(attacked_images)
                
                # 计算损失
                loss, loss_dict = self.loss_fn(
                    watermarked_images, images,
                    decoded_messages, messages,
                    decoded_messages_attacked
                )
                
                # 记录指标
                self.val_metrics.update(**loss_dict)
                
                # 计算额外指标
                psnr = calculate_psnr(watermarked_images, images)
                ssim = calculate_ssim(watermarked_images, images)
                ber = calculate_bitwise_error(decoded_messages, messages)
                
                self.val_metrics.update(psnr=psnr, ssim=ssim, ber=ber)
                
                pbar.set_postfix({
                    'loss': loss.item(),
                    'psnr': psnr,
                    'ber': ber,
                })
        
        # 返回平均损失
        avg_loss = self.val_metrics.get_average('total')
        return avg_loss
    
    def train(self, train_loader, val_loader):
        """
        完整的训练循环
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.config['train']['num_epochs']):
            # 训练
            train_loss = self.train_epoch(train_loader, epoch)
            
            # 验证
            if (epoch + 1) % self.config['eval']['eval_interval'] == 0:
                val_loss = self.validate(val_loader, epoch)
                
                # 记录到TensorBoard
                if self.writer:
                    self.writer.add_scalar('epoch/train_loss', train_loss, epoch)
                    self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
                    self.writer.add_scalar('epoch/train_psnr', 
                                         self.train_metrics.get_average('psnr'), epoch)
                    self.writer.add_scalar('epoch/val_psnr',
                                         self.val_metrics.get_average('psnr'), epoch)
                    self.writer.add_scalar('epoch/train_ber',
                                         self.train_metrics.get_average('ber'), epoch)
                    self.writer.add_scalar('epoch/val_ber',
                                         self.val_metrics.get_average('ber'), epoch)
                
                # 打印指标
                print(f"\nEpoch {epoch+1}/{self.config['train']['num_epochs']}")
                print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                print(f"Train PSNR: {self.train_metrics.get_average('psnr'):.2f}, "
                      f"Val PSNR: {self.val_metrics.get_average('psnr'):.2f}")
                print(f"Train BER: {self.train_metrics.get_average('ber'):.6f}, "
                      f"Val BER: {self.val_metrics.get_average('ber'):.6f}")
                
                # 保存最佳模型
                if self.config['eval']['save_best_model'] and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = epoch
                    self.patience_counter = 0
                    
                    # 保存编码器
                    save_checkpoint(
                        self.encoder,
                        self.encoder_optimizer,
                        epoch,
                        val_loss,
                        self.checkpoint_dir / 'best_encoder.pth'
                    )
                    
                    # 保存解码器
                    save_checkpoint(
                        self.decoder,
                        self.decoder_optimizer,
                        epoch,
                        val_loss,
                        self.checkpoint_dir / 'best_decoder.pth'
                    )
                else:
                    self.patience_counter += 1
                
                # 定期保存检查点
                if (epoch + 1) % self.config['train']['save_interval'] == 0:
                    save_checkpoint(
                        self.encoder,
                        self.encoder_optimizer,
                        epoch,
                        train_loss,
                        self.checkpoint_dir / f'encoder_epoch_{epoch+1}.pth'
                    )
                    save_checkpoint(
                        self.decoder,
                        self.decoder_optimizer,
                        epoch,
                        train_loss,
                        self.checkpoint_dir / f'decoder_epoch_{epoch+1}.pth'
                    )
            
            # 更新学习率
            if epoch >= self.config['train']['warmup_epochs']:
                self.encoder_scheduler.step()
                self.decoder_scheduler.step()
            
            # 早停
            if self.patience_counter >= self.config['eval']['early_stopping_patience']:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # 训练完成
        elapsed_time = time.time() - start_time
        print("\n" + "="*60)
        print(f"Training completed in {format_time(elapsed_time)}")
        print(f"Best model at epoch {self.best_epoch+1} with loss {self.best_val_loss:.6f}")
        print("="*60 + "\n")
        
        if self.writer:
            self.writer.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Train HiDDeN watermarking system')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--tensorboard', action='store_true', help='Enable TensorBoard')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--experiment-name', type=str, default=None, help='Experiment name')
    parser.add_argument('--train-dir', type=str, default='data/train', help='Training data directory')
    parser.add_argument('--val-dir', type=str, default='data/val', help='Validation data directory')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 获取设备
    device = get_device(args.device)
    
    # 获取配置
    config = get_config()
    
    # 更新配置
    config['train']['batch_size'] = args.batch_size
    config['train']['num_epochs'] = args.num_epochs
    config['train']['learning_rate'] = args.learning_rate
    config['log']['tensorboard'] = args.tensorboard
    config['data']['batch_size'] = args.batch_size
    
    # 创建数据加载器
    print("\nLoading data...")
    try:
        train_dataset = WatermarkDataset(
            args.train_dir,
            image_size=config['data']['image_size'],
            message_length=config['model']['message_length'],
            random_crop=True,
        )
        
        val_dataset = WatermarkDataset(
            args.val_dir,
            image_size=config['data']['image_size'],
            message_length=config['model']['message_length'],
            random_crop=False,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config['data']['num_workers'],
            pin_memory=config['data']['pin_memory'],
            drop_last=False,
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Please ensure data directories exist and contain images")
        sys.exit(1)
    
    # 创建训练器
    trainer = Trainer(config, device, args.experiment_name)
    
    # 开始训练
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
