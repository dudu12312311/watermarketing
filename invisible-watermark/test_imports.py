"""
测试所有必要的导入是否正常工作
"""

print("测试导入...")

try:
    print("1. 测试 config...")
    from config import get_config
    print("   ✓ config 导入成功")
    
    print("2. 测试 models...")
    from models import EncoderNet, DecoderNet, NoiseLayerContainer
    print("   ✓ models 导入成功")
    
    print("3. 测试 data...")
    from data import WatermarkDataset
    print("   ✓ data 导入成功")
    
    print("4. 测试 utils...")
    from utils import (
        EncoderLoss, DecoderLoss, CombinedLoss,
        MetricsRecorder, calculate_psnr, calculate_ssim, calculate_bitwise_error,
        set_seed, get_device, save_checkpoint, load_checkpoint,
        count_parameters, print_model_info, create_optimizer, create_scheduler,
        adjust_learning_rate, get_lr, format_time
    )
    print("   ✓ utils 导入成功")
    
    print("\n✅ 所有导入测试通过！")
    print("你现在可以运行训练脚本了：")
    print("python train.py --batch-size 32 --num-epochs 300 --tensorboard")
    
except ImportError as e:
    print(f"\n❌ 导入失败: {e}")
    import traceback
    traceback.print_exc()
