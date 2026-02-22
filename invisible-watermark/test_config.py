"""
测试配置文件是否正确
"""

from config import get_config

print("测试配置文件...")
print("="*60)

try:
    config = get_config()
    
    # 检查所有必要的配置项
    print("\n✓ 数据配置:")
    print(f"  - image_size: {config['data']['image_size']}")
    print(f"  - batch_size: {config['data']['batch_size']}")
    print(f"  - pin_memory: {config['data']['pin_memory']}")
    
    print("\n✓ 训练配置:")
    print(f"  - num_epochs: {config['train']['num_epochs']}")
    print(f"  - learning_rate: {config['train']['learning_rate']}")
    print(f"  - log_interval: {config['train']['log_interval']}")
    print(f"  - save_interval: {config['train']['save_interval']}")
    
    print("\n✓ 日志配置:")
    print(f"  - log_interval: {config['log']['log_interval']}")
    print(f"  - tensorboard: {config['log']['tensorboard']}")
    
    print("\n✓ 模型配置:")
    print(f"  - message_length: {config['model']['message_length']}")
    print(f"  - encoder hidden_channels: {config['model']['encoder']['hidden_channels']}")
    
    print("\n" + "="*60)
    print("✅ 配置文件测试通过！")
    print("="*60)
    
except Exception as e:
    print(f"\n❌ 配置文件错误: {e}")
    import traceback
    traceback.print_exc()
