# HiDDeN隐形水印系统 - 使用指南

## 📋 目录

1. [安装](#安装)
2. [数据准备](#数据准备)
3. [训练模型](#训练模型)
4. [测试模型](#测试模型)
5. [编码图像](#编码图像)
6. [解码图像](#解码图像)
7. [常见问题](#常见问题)

## 安装

### 1. 克隆或下载项目

```bash
cd invisible-watermark
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python test_modules.py
```

如果所有测试通过，说明安装成功。

## 数据准备

### 1. 下载COCO数据集

```bash
# 访问 http://cocodataset.org/#download
# 下载 2017 Train images 和 2017 Val images
```

### 2. 组织数据目录

```
data/
├── train/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── val/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

### 3. 或使用自己的数据集

```bash
# 创建数据目录
mkdir -p data/train data/val

# 将图像放入相应目录
# 支持的格式：JPG, PNG, BMP, TIFF
```

## 训练模型

### 基础训练

```bash
python train.py \
    --batch-size 32 \
    --num-epochs 300 \
    --learning-rate 1e-4 \
    --train-dir data/train \
    --val-dir data/val
```

### 启用TensorBoard

```bash
python train.py \
    --batch-size 32 \
    --num-epochs 300 \
    --tensorboard \
    --train-dir data/train \
    --val-dir data/val

# 在另一个终端查看TensorBoard
tensorboard --logdir logs
```

### 自定义参数

```bash
python train.py \
    --batch-size 64 \
    --num-epochs 500 \
    --learning-rate 5e-5 \
    --device cuda \
    --seed 42 \
    --experiment-name my_experiment \
    --train-dir data/train \
    --val-dir data/val
```

### 训练参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--batch-size` | 32 | 批大小 |
| `--num-epochs` | 300 | 训练轮数 |
| `--learning-rate` | 1e-4 | 学习率 |
| `--device` | cuda | 计算设备 (cuda/cpu) |
| `--tensorboard` | False | 启用TensorBoard |
| `--seed` | 42 | 随机种子 |
| `--experiment-name` | None | 实验名称 |
| `--train-dir` | data/train | 训练数据目录 |
| `--val-dir` | data/val | 验证数据目录 |

### 训练输出

训练完成后，会在以下目录生成文件：

```
logs/
└── exp_YYYYMMDD_HHMMSS/
    └── events.out.tfevents...  # TensorBoard日志

checkpoints/
└── exp_YYYYMMDD_HHMMSS/
    ├── best_encoder.pth        # 最佳编码器
    ├── best_decoder.pth        # 最佳解码器
    ├── encoder_epoch_10.pth    # 第10个epoch的编码器
    └── decoder_epoch_10.pth    # 第10个epoch的解码器
```

## 测试模型

### 基础测试

```bash
python test.py \
    --encoder checkpoints/exp_YYYYMMDD_HHMMSS/best_encoder.pth \
    --decoder checkpoints/exp_YYYYMMDD_HHMMSS/best_decoder.pth \
    --test-dir data/val
```

### 保存测试结果

```bash
python test.py \
    --encoder checkpoints/exp_YYYYMMDD_HHMMSS/best_encoder.pth \
    --decoder checkpoints/exp_YYYYMMDD_HHMMSS/best_decoder.pth \
    --test-dir data/val \
    --output results/test_results.json
```

### 测试参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--encoder` | 必需 | 编码器模型路径 |
| `--decoder` | 必需 | 解码器模型路径 |
| `--test-dir` | data/val | 测试数据目录 |
| `--batch-size` | 32 | 批大小 |
| `--device` | cuda | 计算设备 |
| `--num-workers` | 4 | 数据加载线程数 |
| `--output` | None | 结果输出文件 |

### 测试输出示例

```
================================================================================
Test Results Summary
================================================================================

No Attack:
--------------------------------------------------------------------------------
Metric               Value               
--------------------------------------------------------------------------------
psnr                 38.234567           
ssim                 0.950000            
ber                  0.001234            
mse                  0.012345            

Single Noise Layers:
--------------------------------------------------------------------------------
Noise Type                     PSNR            SSIM            BER            
--------------------------------------------------------------------------------
crop((0.2,0.25),(0.2,0.25))    35.123456       0.920000        0.012345       
cropout((0.55,0.6),(0.55,0.6)) 34.567890       0.910000        0.023456       
dropout(0.55,0.6)              36.234567       0.930000        0.008901       
resize(0.7,0.8)                37.123456       0.940000        0.005678       
jpeg()                         38.012345       0.950000        0.003456       

Combined Noise:
--------------------------------------------------------------------------------
Metric               Value               
--------------------------------------------------------------------------------
psnr                 32.123456           
ssim                 0.880000            
ber                  0.045678            
mse                  0.034567            

================================================================================
```

## 编码图像

### 基础编码

```bash
python encode_image.py \
    --image input.jpg \
    --message "Hello World" \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_encoder.pth \
    --output watermarked.jpg
```

### 编码参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | 必需 | 输入图像路径 |
| `--message` | 必需 | 要隐藏的消息 |
| `--model` | 必需 | 编码器模型路径 |
| `--output` | 必需 | 输出图像路径 |
| `--device` | cuda | 计算设备 |
| `--message-length` | 32 | 消息长度（比特） |

### 编码输出示例

```
Loading image from input.jpg...
Converting message: 'Hello World'
Loading encoder model from checkpoints/exp_YYYYMMDD_HHMMSS/best_encoder.pth...
Encoding image...
Watermarked image saved to watermarked.jpg
PSNR: 38.23 dB

✓ Encoding completed successfully!
```

## 解码图像

### 单个图像解码

```bash
python decode_image.py \
    --image watermarked.jpg \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_decoder.pth \
    --output decoded_message.txt
```

### 批量解码

```bash
python decode_image.py \
    --batch \
    --image-dir watermarked_images/ \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_decoder.pth \
    --output-dir decoded_results/
```

### 解码参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--image` | None | 输入水印图像路径 |
| `--model` | 必需 | 解码器模型路径 |
| `--output` | None | 输出文件路径 |
| `--device` | cuda | 计算设备 |
| `--message-length` | 32 | 消息长度（比特） |
| `--threshold` | 0.5 | 二值化阈值 |
| `--batch` | False | 批量模式 |
| `--image-dir` | None | 批量模式下的图像目录 |
| `--output-dir` | None | 批量模式下的输出目录 |

### 解码输出示例

```
Loading watermarked image from watermarked.jpg...
Loading decoder model from checkpoints/exp_YYYYMMDD_HHMMSS/best_decoder.pth...
Decoding message...

Decoded message: 'Hello World'
Confidence: 0.9876
Raw message (first 32 bits): [0.98, 0.97, 0.02, 0.01, ...]
Results saved to decoded_message.txt

✓ Decoding completed successfully!
```

## 常见问题

### Q1: 显存不足

**A:** 减小批大小或图像大小

```bash
python train.py --batch-size 16 --device cuda
```

或使用CPU训练（较慢）：

```bash
python train.py --batch-size 32 --device cpu
```

### Q2: 训练速度慢

**A:** 增加数据加载线程数或使用更强的GPU

```bash
# 在config.py中修改
DATA_CONFIG = {
    'num_workers': 8,  # 增加线程数
}
```

### Q3: 解码准确率低

**A:** 可能需要更多的训练轮数或调整噪声层参数

```bash
# 增加训练轮数
python train.py --num-epochs 500

# 或调整噪声层配置（在config.py中）
NOISE_CONFIG = {
    'noise_layers': [
        'crop((0.1,0.2),(0.1,0.2))',  # 减小裁剪范围
        'jpeg()',
    ],
}
```

### Q4: 如何使用自定义消息长度

**A:** 在config.py中修改或通过命令行参数

```python
# config.py
MODEL_CONFIG = {
    'message_length': 64,  # 改为64比特
}
```

### Q5: 如何保存和加载模型

**A:** 模型会自动保存到checkpoints目录

```bash
# 加载最佳模型进行推理
python encode_image.py \
    --image input.jpg \
    --message "Test" \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_encoder.pth \
    --output output.jpg
```

### Q6: 如何评估模型性能

**A:** 使用test.py脚本

```bash
python test.py \
    --encoder checkpoints/exp_YYYYMMDD_HHMMSS/best_encoder.pth \
    --decoder checkpoints/exp_YYYYMMDD_HHMMSS/best_decoder.pth \
    --output results/performance.json
```

## 工作流程示例

### 完整的工作流程

```bash
# 1. 准备数据
mkdir -p data/train data/val
# 将图像放入data/train和data/val

# 2. 训练模型
python train.py \
    --batch-size 32 \
    --num-epochs 300 \
    --tensorboard \
    --experiment-name my_first_model

# 3. 测试模型
python test.py \
    --encoder checkpoints/my_first_model/best_encoder.pth \
    --decoder checkpoints/my_first_model/best_decoder.pth \
    --output results/test_results.json

# 4. 编码图像
python encode_image.py \
    --image test_image.jpg \
    --message "Secret Message" \
    --model checkpoints/my_first_model/best_encoder.pth \
    --output watermarked_image.jpg

# 5. 解码图像
python decode_image.py \
    --image watermarked_image.jpg \
    --model checkpoints/my_first_model/best_decoder.pth \
    --output decoded_message.txt

# 6. 查看结果
cat decoded_message.txt
```

## 性能优化建议

### 1. 数据加载优化
- 增加 `num_workers` 以加快数据加载
- 使用 `pin_memory=True` 将数据固定在内存中

### 2. 训练优化
- 使用混合精度训练（需要修改代码）
- 使用梯度累积处理更大的有效批大小
- 使用学习率预热和余弦退火

### 3. 推理优化
- 使用模型量化减小模型大小
- 使用ONNX导出模型以加快推理速度
- 批量处理多个图像

## 更多信息

- 查看 `README.md` 了解项目概述
- 查看 `IMPLEMENTATION_SUMMARY.md` 了解实现细节
- 查看 `config.py` 了解所有可配置参数

---

**最后更新**：2026年2月10日
