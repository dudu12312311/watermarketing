# HiDDeN 隐形水印系统 - 快速开始指南

## 📦 系统已完成

✅ 所有代码实现完成（~3,210行）
✅ 完整的训练框架
✅ 编码/解码功能
✅ 测试评估工具
✅ 详细文档

---

## 🎯 你现在可以做什么

### 1️⃣ 准备数据

```bash
# 创建数据目录
mkdir -p data/train data/val

# 从 http://cocodataset.org/#download 下载COCO 2017
# 放入10,000张训练图像到 data/train/
# 放入1,000张验证图像到 data/val/
```

### 2️⃣ 训练模型

```bash
# 基础训练（无噪声）
python train.py --batch-size 32 --num-epochs 300 --tensorboard

# 或使用特定噪声配置
python train.py --noise 'crop((0.2,0.25),(0.2,0.25))+jpeg()'
```

### 3️⃣ 生成水印图像

```bash
# 编码：隐藏消息到图像
python encode_image.py \
    --image input.jpg \
    --message "Your secret message" \
    --model checkpoints/best_encoder.pth \
    --output watermarked.jpg

# 解码：从水印图像恢复消息
python decode_image.py \
    --image watermarked.jpg \
    --model checkpoints/best_decoder.pth
```

### 4️⃣ 测试系统

```bash
# 运行完整测试
python test.py --model checkpoints/best_model.pth
```

---

## 📁 项目结构

```
invisible-watermark/
├── models/              # 神经网络模型
│   ├── encoder.py       # 编码器
│   ├── decoder.py       # 解码器
│   └── noise_layers.py  # 噪声层
├── data/
│   └── dataset.py       # 数据加载
├── utils/
│   ├── metrics.py       # 评估指标
│   ├── losses.py        # 损失函数
│   └── helpers.py       # 辅助函数
├── train.py             # 训练脚本
├── test.py              # 测试脚本
├── encode_image.py      # 编码脚本
├── decode_image.py      # 解码脚本
├── config.py            # 配置文件
└── requirements.txt     # 依赖
```

---

## 🔧 配置参数

编辑 `config.py` 自定义参数：

```python
# 数据配置
DATA_CONFIG = {
    'image_size': 400,      # 图像大小
    'batch_size': 32,       # 批大小
    'num_workers': 4,       # 数据加载线程
}

# 模型配置
MODEL_CONFIG = {
    'message_length': 32,   # 消息长度（比特）
    'hidden_channels': 64,  # 隐藏层通道数
}

# 训练配置
TRAIN_CONFIG = {
    'num_epochs': 300,      # 训练轮数
    'learning_rate': 1e-4,  # 学习率
    'batch_size': 32,       # 批大小
}
```

---

## 📊 性能指标

### 无噪声
- Encoder MSE: < 0.01
- PSNR: > 40dB
- Bitwise Error: < 0.001

### 单个噪声层
- Encoder MSE: < 0.05
- Bitwise Error: < 0.01
- PSNR: > 35dB

### 组合噪声层
- Encoder MSE: < 0.1
- Bitwise Error: < 0.05
- PSNR: > 30dB

---

## 🎯 噪声层配置

### 支持的噪声类型

| 噪声类型 | 配置示例 | 说明 |
|---------|---------|------|
| Crop | `crop((0.2,0.25),(0.2,0.25))` | 随机裁剪 |
| Cropout | `cropout((0.55,0.6),(0.55,0.6))` | 随机删除区域 |
| Dropout | `dropout(0.55,0.6)` | 随机丢弃像素 |
| Resize | `resize(0.7,0.8)` | 随机缩放 |
| JPEG | `jpeg()` | JPEG压缩 |

### 组合示例

```bash
# 单个噪声
python train.py --noise 'crop((0.2,0.25),(0.2,0.25))'

# 多个噪声（随机选择一个）
python train.py --noise 'crop((0.2,0.25),(0.2,0.25))+jpeg()+resize(0.7,0.8)'
```

---

## 📚 文档

- **README.md** - 项目概述
- **USAGE_GUIDE.md** - 详细使用指南
- **NEXT_STEPS.md** - 后续工作步骤
- **IMPLEMENTATION_SUMMARY.md** - 实现细节
- **PROJECT_COMPLETION_REPORT.md** - 完成报告

---

## ⚡ 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 基础训练
python train.py

# 自定义训练
python train.py \
    --batch-size 32 \
    --num-epochs 300 \
    --learning-rate 1e-4 \
    --noise 'crop((0.2,0.25),(0.2,0.25))+jpeg()' \
    --tensorboard

# 编码单张图像
python encode_image.py \
    --image input.jpg \
    --message "Hello" \
    --model checkpoints/best_encoder.pth \
    --output output.jpg

# 解码单张图像
python decode_image.py \
    --image output.jpg \
    --model checkpoints/best_decoder.pth

# 运行测试
python test.py --model checkpoints/best_model.pth

# 查看所有参数
python train.py --help
```

---

## 🚀 下一步

1. **准备数据** - 下载COCO数据集
2. **训练模型** - 运行训练脚本
3. **生成水印** - 编码图像
4. **测试系统** - 验证性能

详见 `NEXT_STEPS.md`

---

## 💡 提示

- 首次运行建议使用小数据集测试
- 使用 `--tensorboard` 参数可视化训练过程
- 模型检查点自动保存在 `checkpoints/` 目录
- 所有实验结果保存在 `runs/` 目录

---

## 📞 获取帮助

查看详细文档：
- 使用问题 → `USAGE_GUIDE.md`
- 实现细节 → `IMPLEMENTATION_SUMMARY.md`
- 后续步骤 → `NEXT_STEPS.md`
- 完成报告 → `PROJECT_COMPLETION_REPORT.md`

---

**准备好开始了吗？** 👉 查看 `NEXT_STEPS.md` 了解详细步骤！
