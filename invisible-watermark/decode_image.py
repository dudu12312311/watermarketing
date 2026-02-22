"""
图像解码脚本 - 从水印图像中恢复隐藏的消息
"""

import os
import argparse
import torch
import numpy as np
from PIL import Image
from pathlib import Path

from config import get_config
from models import DecoderNet
from utils.helpers import get_device, load_model_weights


def binary_to_text(binary_message, threshold=0.5):
    """
    将二进制消息转换为文本
    
    Args:
        binary_message: 二进制消息（0-1之间的浮点数）
        threshold: 二值化阈值
    
    Returns:
        text: 解码的文本
    """
    # 二值化
    binary = (binary_message > threshold).astype(int)
    
    # 转换为字节
    text_bytes = []
    for i in range(0, len(binary), 8):
        if i + 8 <= len(binary):
            byte = 0
            for j in range(8):
                byte |= binary[i + j] << j
            if byte != 0:  # 忽略空字节
                text_bytes.append(byte)
    
    # 转换为文本
    try:
        text = bytes(text_bytes).decode('utf-8', errors='ignore')
    except:
        text = ""
    
    return text


def decode_image(image_path, model_path, output_path=None, device='cuda', message_length=32, threshold=0.5):
    """
    解码图像
    
    Args:
        image_path: 输入水印图像路径
        model_path: 模型权重路径
        output_path: 输出文件路径（可选）
        device: 计算设备
        message_length: 消息长度
        threshold: 二值化阈值
    
    Returns:
        message: 恢复的消息
        confidence: 置信度
    """
    # 获取配置
    config = get_config()
    
    # 加载图像
    print(f"Loading watermarked image from {image_path}...")
    image = Image.open(image_path).convert('RGB')
    
    # 调整大小
    image_size = config['data']['image_size']
    image = image.resize((image_size, image_size), Image.BILINEAR)
    
    # 转换为张量
    image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 127.5 - 1.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    
    # 加载模型
    print(f"Loading decoder model from {model_path}...")
    decoder = DecoderNet(
        message_length=config['model']['message_length'],
        hidden_channels=config['model']['decoder']['hidden_channels'],
        num_layers=config['model']['decoder']['num_layers'],
    ).to(device)
    
    load_model_weights(decoder, model_path, device)
    decoder.eval()
    
    # 解码
    print("Decoding message...")
    with torch.no_grad():
        message = decoder(image_tensor)
    
    # 转换为numpy
    message_np = message.squeeze(0).cpu().numpy()
    
    # 计算置信度（消息与0.5的距离）
    confidence = 1.0 - np.mean(np.abs(message_np - 0.5)) * 2
    
    # 转换为文本
    text = binary_to_text(message_np, threshold)
    
    # 打印结果
    print(f"\nDecoded message: '{text}'")
    print(f"Confidence: {confidence:.4f}")
    print(f"Raw message (first 32 bits): {message_np[:32]}")
    
    # 保存到文件
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Decoded Message: {text}\n")
            f.write(f"Confidence: {confidence:.4f}\n")
            f.write(f"Raw Message: {message_np.tolist()}\n")
        print(f"Results saved to {output_path}")
    
    return text, confidence, message_np


def decode_batch(image_dir, model_path, output_dir=None, device='cuda', message_length=32):
    """
    批量解码图像
    
    Args:
        image_dir: 图像目录
        model_path: 模型权重路径
        output_dir: 输出目录
        device: 计算设备
        message_length: 消息长度
    """
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(image_dir).glob(ext))
        image_files.extend(Path(image_dir).glob(ext.upper()))
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 解码每个图像
    results = []
    for image_path in image_files:
        print(f"\nProcessing {image_path.name}...")
        
        try:
            output_file = None
            if output_dir:
                output_file = os.path.join(output_dir, f"{image_path.stem}_decoded.txt")
            
            text, confidence, _ = decode_image(
                str(image_path),
                model_path,
                output_file,
                device,
                message_length
            )
            
            results.append({
                'image': image_path.name,
                'message': text,
                'confidence': confidence,
            })
        except Exception as e:
            print(f"Error processing {image_path.name}: {e}")
    
    # 打印总结
    print("\n" + "="*60)
    print("Batch Decoding Results")
    print("="*60)
    for result in results:
        print(f"{result['image']}: '{result['message']}' (confidence: {result['confidence']:.4f})")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Decode message from watermarked image')
    parser.add_argument('--image', type=str, help='Input watermarked image path')
    parser.add_argument('--model', type=str, required=True, help='Decoder model path')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--message-length', type=int, default=32, help='Message length in bits')
    parser.add_argument('--threshold', type=float, default=0.5, help='Binarization threshold')
    parser.add_argument('--batch', action='store_true', help='Batch mode')
    parser.add_argument('--image-dir', type=str, help='Image directory for batch mode')
    parser.add_argument('--output-dir', type=str, help='Output directory for batch mode')
    
    args = parser.parse_args()
    
    # 获取设备
    device = get_device(args.device)
    
    try:
        if args.batch:
            # 批量模式
            if not args.image_dir:
                print("Error: --image-dir is required for batch mode")
                return
            
            decode_batch(
                args.image_dir,
                args.model,
                args.output_dir,
                device,
                args.message_length
            )
        else:
            # 单个图像模式
            if not args.image:
                print("Error: --image is required for single image mode")
                return
            
            decode_image(
                args.image,
                args.model,
                args.output,
                device,
                args.message_length,
                args.threshold
            )
        
        print("\n✓ Decoding completed successfully!")
    except Exception as e:
        print(f"\n✗ Error during decoding: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
