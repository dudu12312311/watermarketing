"""
从大数据集中选择部分图片创建小数据集
"""

import os
import shutil
import random
from pathlib import Path

def prepare_small_dataset(
    source_train_dir='data/train',
    source_val_dir='data/val',
    target_train_dir='data/small/train',
    target_val_dir='data/small/val',
    num_train=500,
    num_val=100
):
    """
    从大数据集中随机选择图片创建小数据集
    
    Args:
        source_train_dir: 源训练集目录（10,000张图片）
        source_val_dir: 源验证集目录（1,000张图片）
        target_train_dir: 目标训练集目录（500张图片）
        target_val_dir: 目标验证集目录（100张图片）
        num_train: 训练集图片数量
        num_val: 验证集图片数量
    """
    
    print("="*60)
    print("准备小数据集")
    print("="*60)
    
    # 创建目标目录
    print(f"\n步骤1: 创建目标目录...")
    os.makedirs(target_train_dir, exist_ok=True)
    os.makedirs(target_val_dir, exist_ok=True)
    print(f"  ✓ 创建 {target_train_dir}")
    print(f"  ✓ 创建 {target_val_dir}")
    
    # 获取源图片列表
    print(f"\n步骤2: 扫描源数据集...")
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
    
    # 获取训练集图片
    train_images = []
    for ext in image_extensions:
        train_images.extend(Path(source_train_dir).glob(f'*{ext}'))
    train_images = [str(img) for img in train_images]
    
    # 获取验证集图片
    val_images = []
    for ext in image_extensions:
        val_images.extend(Path(source_val_dir).glob(f'*{ext}'))
    val_images = [str(img) for img in val_images]
    
    print(f"  ✓ 找到 {len(train_images)} 张训练图片")
    print(f"  ✓ 找到 {len(val_images)} 张验证图片")
    
    # 检查是否有足够的图片
    if len(train_images) < num_train:
        print(f"\n  ⚠️  警告: 训练集只有 {len(train_images)} 张图片，少于需要的 {num_train} 张")
        num_train = len(train_images)
        print(f"  → 将使用所有 {num_train} 张训练图片")
    
    if len(val_images) < num_val:
        print(f"\n  ⚠️  警告: 验证集只有 {len(val_images)} 张图片，少于需要的 {num_val} 张")
        num_val = len(val_images)
        print(f"  → 将使用所有 {num_val} 张验证图片")
    
    # 随机选择图片
    print(f"\n步骤3: 随机选择图片...")
    random.seed(42)  # 设置随机种子，保证可重复
    selected_train = random.sample(train_images, num_train)
    selected_val = random.sample(val_images, num_val)
    print(f"  ✓ 选择了 {len(selected_train)} 张训练图片")
    print(f"  ✓ 选择了 {len(selected_val)} 张验证图片")
    
    # 复制训练集图片
    print(f"\n步骤4: 复制训练集图片...")
    for i, img_path in enumerate(selected_train, 1):
        filename = os.path.basename(img_path)
        target_path = os.path.join(target_train_dir, filename)
        shutil.copy2(img_path, target_path)
        if i % 50 == 0 or i == len(selected_train):
            print(f"  进度: {i}/{len(selected_train)} ({i*100//len(selected_train)}%)")
    print(f"  ✓ 完成训练集复制")
    
    # 复制验证集图片
    print(f"\n步骤5: 复制验证集图片...")
    for i, img_path in enumerate(selected_val, 1):
        filename = os.path.basename(img_path)
        target_path = os.path.join(target_val_dir, filename)
        shutil.copy2(img_path, target_path)
        if i % 20 == 0 or i == len(selected_val):
            print(f"  进度: {i}/{len(selected_val)} ({i*100//len(selected_val)}%)")
    print(f"  ✓ 完成验证集复制")
    
    # 验证结果
    print(f"\n步骤6: 验证结果...")
    train_count = len(list(Path(target_train_dir).glob('*.*')))
    val_count = len(list(Path(target_val_dir).glob('*.*')))
    print(f"  ✓ 训练集: {train_count} 张图片")
    print(f"  ✓ 验证集: {val_count} 张图片")
    
    # 计算磁盘占用
    train_size = sum(os.path.getsize(f) for f in Path(target_train_dir).glob('*.*')) / (1024*1024)
    val_size = sum(os.path.getsize(f) for f in Path(target_val_dir).glob('*.*')) / (1024*1024)
    total_size = train_size + val_size
    print(f"  ✓ 磁盘占用: {total_size:.1f} MB")
    
    print("\n" + "="*60)
    print("✅ 小数据集准备完成！")
    print("="*60)
    print(f"\n数据集位置:")
    print(f"  训练集: {target_train_dir}")
    print(f"  验证集: {target_val_dir}")
    print(f"\n下一步:")
    print(f"  运行训练命令:")
    print(f"  python train.py --train-dir {target_train_dir} --val-dir {target_val_dir} --batch-size 16 --num-epochs 100 --tensorboard")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='准备小数据集')
    parser.add_argument('--source-train', type=str, default='data/train', 
                        help='源训练集目录（默认: data/train）')
    parser.add_argument('--source-val', type=str, default='data/val',
                        help='源验证集目录（默认: data/val）')
    parser.add_argument('--target-train', type=str, default='data/small/train',
                        help='目标训练集目录（默认: data/small/train）')
    parser.add_argument('--target-val', type=str, default='data/small/val',
                        help='目标验证集目录（默认: data/small/val）')
    parser.add_argument('--num-train', type=int, default=500,
                        help='训练集图片数量（默认: 500）')
    parser.add_argument('--num-val', type=int, default=100,
                        help='验证集图片数量（默认: 100）')
    
    args = parser.parse_args()
    
    try:
        prepare_small_dataset(
            source_train_dir=args.source_train,
            source_val_dir=args.source_val,
            target_train_dir=args.target_train,
            target_val_dir=args.target_val,
            num_train=args.num_train,
            num_val=args.num_val
        )
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
