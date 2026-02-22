"""
图像编码脚本 - 将消息隐藏到图像中
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from config import get_config
from models import EncoderNet
from utils.helpers import get_device, load_model_weights


def text_to_binary(text, message_length=32):
    """
    将文本转换为二进制消息
    
    Args:
        text: 输入文本
        message_length: 消息长度（比特数）
    
    Returns:
        message: 二进制消息张量
    """
    # 将文本转换为字节
    text_bytes = text.encode('utf-8')
    
    # 创建二进制消息
    message = []
    for byte in text_bytes:
        for i in range(8):
            message.append((byte >> i) & 1)
    
    # 填充或截断到指定长度
    if len(message) < message_length:
        message.extend([0] * (message_length - len(message)))
    else:
        message = message[:message_length]
    
    return torch.tensor(message, dtype=torch.float32)


def encode_image(image_path, message, model_path, output_path, device='cuda', message_length=32):
    """
    编码图像
    
    Args:
        image_path: 输入图像路径
        message: 消息（文本或二进制）
        model_path: 模型权重路径
        output_path: 输出图像路径
        device: 计算设备
        message_length: 消息长度
    """
    # 获取配置
    config = get_config()
    
    # 加载图像
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert('RGB')
    
    # 调整大小
    image_size = config['data']['image_size']
    image = image.resize((image_size, image_size), Image.BILINEAR)
    
    # 转换为张量
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 127.5 - 1.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 处理消息
    if isinstance(message, str):
        print(f"Converting message: '{message}'")
        message_tensor = text_to_binary(message, message_length).unsqueeze(0).to(device)
    else:
        message_tensor = message.unsqueeze(0).to(device)
    
    # 加载模型
    print(f"Loading encoder model from {model_path}...")
    encoder = EncoderNet(
        message_length=config['model']['message_length'],
        hidden_channels=config['model']['encoder']['hidden_channels'],
        num_layers=config['model']['encoder']['num_layers'],
    ).to(device)
    
    load_model_weights(encoder, model_path, device)
    encoder.eval()
    
    # 编码
    print("Encoding image...")
    with torch.no_grad():
        watermarked_image = encoder(image_tensor, message_tensor)
    
    # 转换回图像
    watermarked_image = watermarked_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    watermarked_image = ((watermarked_image + 1.0) * 127.5).astype(np.uint8)
    watermarked_image = Image.fromarray(watermarked_image)
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    watermarked_image.save(output_path)
    print(f"Watermarked image saved to {output_path}")
    
    # 计算PSNR
    original_np = np.array(image, dtype=np.float32) / 255.0
    watermarked_np = np.array(watermarked_image, dtype=np.float32) / 255.0
    mse = np.mean((original_np - watermarked_np) ** 2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    print(f"PSNR: {psnr:.2f} dB")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Encode message into image')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--message', type=str, required=True, help='Message to hide')
    parser.add_argument('--model', type=str, required=True, help='Encoder model path')
    parser.add_argument('--output', type=str, required=True, help='Output image path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--message-length', type=int, default=32, help='Message length in bits')
    
    args = parser.parse_args()
    
    # 获取设备
    device = get_device(args.device)
    
    # 编码图像
    try:
        encode_image(
            args.image,
            args.message,
            args.model,
            args.output,
            device,
            args.message_length
        )
        print("\n✓ Encoding completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during encoding: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
