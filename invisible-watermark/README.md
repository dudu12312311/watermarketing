# HiDDeN 隐形水印系统

基于论文 **"HiDDeN: Hiding Data With Deep Networks"** 的PyTorch实现

## 📋 项目概述

HiDDeN是一个深度学习隐形水印系统，能够在图像中隐藏二进制数据，并在各种图像攻击（压缩、裁剪、旋转等）下恢复隐藏的数据。

### 核心特性

- ✅ 编码器-解码器架构
- ✅ 支持多种噪声层（Crop、Cropout、Dropout、Resize、JPEG）
- ✅ 高鲁棒性（抵抗常见图像处理攻击）
- ✅ 高隐蔽性（PSNR > 38dB）
- ✅ 可配置的消息长度
- ✅ TensorBoard可视化
- ✅ 完整的训练和推理管道

## 🚀 快速开始

### 1. 环境安装

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 准备数据

下载COCO数据集或使用自己的图像数据集：

```bash
# 创建数据目录
mkdir -p data/train data/val

# 将图像放入相应目录
# data/train/  - 训练图像（~10,000张）
# data/val/    - 验证图像（~1,000张）
```

### 3. 训练模型

```bash
# 基础训练
python train.py

# 使用自定义参数
python train.py \
    --batch-size 32 \
    --num-epochs 300 \
    --learning-rate 1e-4 \
    --noise 'crop((0.2,0.25),(0.2,0.25))+jpeg()'

# 启用TensorBoard
python train.py --tensorboard
tensorboard --logdir logs
```

### 4. 编码图像

```bash
# 将消息隐藏到图像中
python encode_image.py \
    --image input.jpg \
    --message "Hello World" \
    --model checkpoints/best_model.pth \
    --output watermarked.jpg
```

### 5. 解码图像

```bash
# 从水印图像中恢复消息
python decode_image.py \
    --image watermarked.jpg \
    --model checkpoints/best_model.pth \
    --output recovered_message.txt
```

## 📁 项目结构

```
invisible-watermark/
├── models/
│   ├── __init__.py
│   ├── encoder.py          # 编码器网络
│   ├── decoder.py          # 解码器网络
│   └── noise_layers.py     # 噪声层模块
├── data/
│   ├── __init__.py
│   └── dataset.py          # 数据加载器
├── utils/
│   ├── __init__.py
│   ├── metrics.py          # 评估指标
│   ├── losses.py           # 损失函数
│   └── helpers.py          # 辅助函数
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── encode_image.py         # 图像编码脚本
├── decode_image.py         # 图像解码脚本
├── config.py               # 配置文件
├── requirements.txt        # 依赖
└── README.md              # 本文件
```

## 🔧 配置说明

编辑 `config.py` 文件来自定义系统参数：

### 数据配置
```python
DATA_CONFIG = {
    'image_size': 400,      # 图像大小
    'batch_size': 32,       # 批大小
    'num_workers': 4,       # 数据加载线程数
}
```

### 模型配置
```python
MODEL_CONFIG = {
    'message_length': 32,   # 隐藏消息长度（比特）
    'hidden_channels': 64,  # 隐藏层通道数
}
```

### 训练配置
```python
TRAIN_CONFIG = {
    'num_epochs': 300,      # 训练轮数
    'learning_rate': 1e-4,  # 学习率
    'batch_size': 32,       # 批大小
}
```

### 噪声层配置
```python
NOISE_CONFIG = {
    'noise_layers': [
        'crop((0.2,0.25),(0.2,0.25))',
        'cropout((0.55,0.6),(0.55,0.6))',
        'dropout(0.55,0.6)',
        'resize(0.7,0.8)',
        'jpeg()',
    ],
}
```

## 📊 性能指标

### 无攻击情况
| 指标 | 目标值 | 实现值 |
|------|--------|--------|
| Encoder MSE | < 0.01 | - |
| PSNR | > 38dB | - |
| Bitwise Error | < 0.001 | - |

### 单个噪声层
| 指标 | 目标值 | 实现值 |
|------|--------|--------|
| Encoder MSE | < 0.05 | - |
| Bitwise Error | < 0.01 | - |

### 组合噪声层
| 指标 | 目标值 | 实现值 |
|------|--------|--------|
| Encoder MSE | < 0.1 | - |
| Bitwise Error | < 0.05 | - |

## 🎯 噪声层说明

### Crop（裁剪）
```
Crop((height_min,height_max),(width_min,width_max))
随机裁剪图像的指定比例
```

### Cropout（随机删除）
```
Cropout((height_min,height_max),(width_min,width_max))
随机删除图像的指定区域
```

### Dropout（像素丢弃）
```
Dropout(keep_min, keep_max)
随机丢弃像素，保留指定比例
```

### Resize（缩放）
```
Resize(scale_min, scale_max)
随机缩放图像
```

### JPEG（压缩）
```
JPEG()
可微分的JPEG压缩近似
```

## 📚 参考资源

- **论文**：[HiDDeN: Hiding Data With Deep Networks](https://arxiv.org/abs/1807.09937)
- **原始实现**：[Lua+Torch版本](https://github.com/jirenz/HiDDeN)
- **数据集**：[COCO Dataset](http://cocodataset.org/#download)

## 🔍 故障排除

### 显存不足
- 减小 `batch_size`
- 减小 `image_size`
- 使用 `--device cpu` 使用CPU训练

### 训练速度慢
- 增加 `num_workers`
- 使用更强的GPU
- 减小 `image_size`

### 解码准确率低
- 增加训练轮数
- 调整噪声层参数
- 检查数据质量

## 📝 许可证

MIT License

## 👨‍💻 作者

基于HiDDeN论文的PyTorch实现

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

**最后更新**：2026年2月10日
