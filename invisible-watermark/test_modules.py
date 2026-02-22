"""
模块测试脚本 - 验证所有核心模块都能正常工作
"""

import torch
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from models import EncoderNet, DecoderNet, NoiseLayerContainer
from utils import MetricsRecorder, EncoderLoss, DecoderLoss
from utils.helpers import set_seed, get_device, count_parameters, print_model_info


def test_encoder():
    """测试编码器"""
    print("\n" + "="*60)
    print("Testing EncoderNet...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = EncoderNet(message_length=32, hidden_channels=64, num_layers=4).to(device)
    
    # 打印模型信息
    print_model_info(encoder, "EncoderNet")
    
    # 创建测试输入
    batch_size = 2
    image = torch.randn(batch_size, 3, 400, 400).to(device)
    message = torch.randint(0, 2, (batch_size, 32)).float().to(device)
    
    # 前向传播
    watermarked = encoder(image, message)
    
    print(f"✓ Input image shape: {image.shape}")
    print(f"✓ Message shape: {message.shape}")
    print(f"✓ Watermarked image shape: {watermarked.shape}")
    print(f"✓ Watermarked image range: [{watermarked.min():.4f}, {watermarked.max():.4f}]")
    
    # 计算MSE
    mse = torch.mean((watermarked - image) ** 2)
    print(f"✓ Encoder MSE: {mse.item():.6f}")
    
    return True


def test_decoder():
    """测试解码器"""
    print("\n" + "="*60)
    print("Testing DecoderNet...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decoder = DecoderNet(message_length=32, hidden_channels=64, num_layers=4).to(device)
    
    # 打印模型信息
    print_model_info(decoder, "DecoderNet")
    
    # 创建测试输入
    batch_size = 2
    watermarked_image = torch.randn(batch_size, 3, 400, 400).to(device)
    
    # 前向传播
    message = decoder(watermarked_image)
    
    print(f"✓ Watermarked image shape: {watermarked_image.shape}")
    print(f"✓ Recovered message shape: {message.shape}")
    print(f"✓ Message range: [{message.min():.4f}, {message.max():.4f}]")
    
    return True


def test_noise_layers():
    """测试噪声层"""
    print("\n" + "="*60)
    print("Testing Noise Layers...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试图像
    image = torch.randn(2, 3, 400, 400).to(device)
    
    # 测试各个噪声层
    noise_configs = [
        'crop((0.2,0.25),(0.2,0.25))',
        'cropout((0.55,0.6),(0.55,0.6))',
        'dropout(0.55,0.6)',
        'resize(0.7,0.8)',
        'jpeg()',
        'crop((0.2,0.25),(0.2,0.25))+jpeg()',
    ]
    
    for config in noise_configs:
        try:
            container = NoiseLayerContainer(config, use_random=False).to(device)
            result = container(image)
            print(f"✓ {config}: {image.shape} -> {result.shape}")
        except Exception as e:
            print(f"✗ {config}: {e}")
            return False
    
    return True


def test_losses():
    """测试损失函数"""
    print("\n" + "="*60)
    print("Testing Loss Functions...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 2
    message_length = 32
    
    watermarked = torch.randn(batch_size, 3, 100, 100).to(device)
    original = torch.randn(batch_size, 3, 100, 100).to(device)
    decoded_msg = torch.rand(batch_size, message_length).to(device)
    original_msg = torch.randint(0, 2, (batch_size, message_length)).float().to(device)
    
    # 测试编码器损失
    try:
        encoder_loss_fn = EncoderLoss(mse_weight=1.0, bce_weight=1.0)
        loss, loss_dict = encoder_loss_fn(watermarked, original, decoded_msg, original_msg)
        print(f"✓ EncoderLoss: {loss.item():.6f}")
    except Exception as e:
        print(f"✗ EncoderLoss: {e}")
        return False
    
    # 测试解码器损失
    try:
        decoder_loss_fn = DecoderLoss()
        loss = decoder_loss_fn(decoded_msg, original_msg)
        print(f"✓ DecoderLoss: {loss.item():.6f}")
    except Exception as e:
        print(f"✗ DecoderLoss: {e}")
        return False
    
    return True


def test_metrics():
    """测试指标"""
    print("\n" + "="*60)
    print("Testing Metrics...")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    img1 = torch.randn(2, 3, 100, 100).to(device)
    img2 = img1 + torch.randn_like(img1) * 0.01
    
    # 测试指标记录器
    try:
        recorder = MetricsRecorder()
        for i in range(10):
            recorder.update(
                loss=torch.tensor(0.5 - i * 0.01),
                psnr=torch.tensor(30 + i),
                ssim=torch.tensor(0.8 + i * 0.01),
            )
        
        averages = recorder.get_all_averages()
        print(f"✓ MetricsRecorder:")
        for key, value in averages.items():
            print(f"  - {key}: {value:.6f}")
    except Exception as e:
        print(f"✗ MetricsRecorder: {e}")
        return False
    
    return True


def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("HiDDeN Watermarking System - Module Test")
    print("="*60)
    
    # 设置随机种子
    set_seed(42)
    
    # 获取设备
    device = get_device('cuda')
    
    # 运行测试
    tests = [
        ("Encoder", test_encoder),
        ("Decoder", test_decoder),
        ("Noise Layers", test_noise_layers),
        ("Loss Functions", test_losses),
        ("Metrics", test_metrics),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n✗ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # 打印总结
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    # 总体结果
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
