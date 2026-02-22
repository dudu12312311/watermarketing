# 🎓 完整操作指南 - 从数据准备到训练完成

## 📋 当前状态

你已经有：
- ✅ `data/train/` 文件夹：10,000张图片
- ✅ `data/val/` 文件夹：1,000张图片

你想要：
- 🎯 使用 500张训练图片 + 100张验证图片

---

## 🚀 完整操作步骤

### 第一步：准备小数据集（从10,000张中选500张）

#### 1.1 打开命令提示符（CMD）

1. 按键盘 `Win + R`
2. 输入 `cmd`
3. 按回车

#### 1.2 进入项目目录

在CMD中输入（把路径改成你的实际路径）：

```cmd
cd D:\大创 视觉\李威\李威\yolov5-6.2\invisible-watermark
```

按回车。

#### 1.3 运行数据准备脚本

在CMD中输入：

```cmd
python prepare_small_dataset.py
```

按回车。

**你会看到：**
```
============================================================
准备小数据集
============================================================

步骤1: 创建目标目录...
  ✓ 创建 data/small/train
  ✓ 创建 data/small/val

步骤2: 扫描源数据集...
  ✓ 找到 10000 张训练图片
  ✓ 找到 1000 张验证图片

步骤3: 随机选择图片...
  ✓ 选择了 500 张训练图片
  ✓ 选择了 100 张验证图片

步骤4: 复制训练集图片...
  进度: 50/500 (10%)
  进度: 100/500 (20%)
  进度: 150/500 (30%)
  进度: 200/500 (40%)
  进度: 250/500 (50%)
  进度: 300/500 (60%)
  进度: 350/500 (70%)
  进度: 400/500 (80%)
  进度: 450/500 (90%)
  进度: 500/500 (100%)
  ✓ 完成训练集复制

步骤5: 复制验证集图片...
  进度: 20/100 (20%)
  进度: 40/100 (40%)
  进度: 60/100 (60%)
  进度: 80/100 (80%)
  进度: 100/100 (100%)
  ✓ 完成验证集复制

步骤6: 验证结果...
  ✓ 训练集: 500 张图片
  ✓ 验证集: 100 张图片
  ✓ 磁盘占用: 150.5 MB

============================================================
✅ 小数据集准备完成！
============================================================
```

**这个过程需要：** 约1-2分钟

---

### 第二步：验证数据集

#### 2.1 检查文件夹是否创建成功

在CMD中输入：

```cmd
dir data\small\train
```

你应该看到500个图片文件。

再输入：

```cmd
dir data\small\val
```

你应该看到100个图片文件。

#### 2.2 验证图片数量

在CMD中输入：

```cmd
dir data\small\train /b | find /c /v ""
```

应该显示：`500`

再输入：

```cmd
dir data\small\val /b | find /c /v ""
```

应该显示：`100`

**如果数字正确，说明数据准备成功！** ✅

---

### 第三步：开始训练

#### 3.1 确认你还在正确的目录

在CMD中，确保你在 `invisible-watermark` 目录下。如果不确定，重新输入：

```cmd
cd D:\大创 视觉\李威\李威\yolov5-6.2\invisible-watermark
```

#### 3.2 运行训练命令

在CMD中输入（这是一行命令，用 `^` 符号换行）：

```cmd
python train.py --train-dir data/small/train --val-dir data/small/val --batch-size 16 --num-epochs 100 --tensorboard
```

按回车。

**命令解释：**
- `python train.py` - 运行训练脚本
- `--train-dir data/small/train` - 使用小训练集（500张）
- `--val-dir data/small/val` - 使用小验证集（100张）
- `--batch-size 16` - 每次处理16张图片（节省内存）
- `--num-epochs 100` - 训练100轮（约2-3小时）
- `--tensorboard` - 启用可视化监控

---

### 第四步：观察训练过程

#### 4.1 训练开始后你会看到

```
Loading data...
Training samples: 500
Validation samples: 100
Experiment directory: logs/exp_20260210_235959
Checkpoint directory: checkpoints/exp_20260210_235959

EncoderNet:
  Parameters: 1,234,567
  Size: 4.71 MB

DecoderNet:
  Parameters: 987,654
  Size: 3.77 MB

============================================================
Starting Training
============================================================

Epoch 1/100: 100%|████████| 31/31 [00:45<00:00, 1.45s/it, loss=0.0234, psnr=38.5, ber=0.0012]

Epoch 1/100
Train Loss: 0.023456, Val Loss: 0.025678
Train PSNR: 38.50, Val PSNR: 37.20
Train BER: 0.001234, Val BER: 0.001567
```

#### 4.2 理解输出信息

- **loss（损失）**：越小越好，目标 < 0.01
- **psnr（峰值信噪比）**：越大越好，目标 > 35dB
- **ber（比特错误率）**：越小越好，目标 < 0.01

#### 4.3 训练时间估计

| 硬件配置 | 预计时间 |
|---------|---------|
| NVIDIA GPU（如RTX 3060） | 1-2小时 |
| NVIDIA GPU（如GTX 1060） | 2-3小时 |
| CPU（Intel i7） | 8-12小时 |
| CPU（Intel i5） | 12-16小时 |

**建议：** 让电脑插着电源，不要休眠。

---

### 第五步：监控训练进度（可选）

#### 5.1 打开新的CMD窗口

1. 再次按 `Win + R`
2. 输入 `cmd`
3. 按回车

#### 5.2 启动TensorBoard

在新的CMD窗口中输入：

```cmd
cd D:\大创 视觉\李威\李威\yolov5-6.2\invisible-watermark
tensorboard --logdir logs
```

#### 5.3 打开浏览器

在浏览器中访问：

```
http://localhost:6006
```

你会看到训练曲线的实时图表。

---

### 第六步：等待训练完成

#### 6.1 训练完成的标志

当你看到：

```
============================================================
Training completed in 2h 15m 30s
Best model at epoch 87 with loss 0.008234
============================================================
```

说明训练完成了！

#### 6.2 检查模型文件

在CMD中输入：

```cmd
dir checkpoints
```

你应该看到一个新文件夹，名字类似 `exp_20260210_235959`

进入这个文件夹：

```cmd
dir checkpoints\exp_20260210_235959
```

你应该看到：
- `best_encoder.pth` - 最佳编码器模型
- `best_decoder.pth` - 最佳解码器模型

**这些就是训练好的模型！** 🎉

---

### 第七步：测试模型（编码图像）

#### 7.1 准备测试图片

1. 创建测试文件夹：

```cmd
mkdir test_images
mkdir watermarked_images
```

2. 把一张你想测试的图片（JPG或PNG格式）放到 `test_images` 文件夹
3. 假设图片名字是 `test.jpg`

#### 7.2 编码图像（隐藏消息）

在CMD中输入（记得替换模型路径中的时间戳）：

```cmd
python encode_image.py --image test_images/test.jpg --message "Hello World" --model checkpoints/exp_20260210_235959/best_encoder.pth --output watermarked_images/test_watermarked.jpg
```

**你会看到：**
```
Loading image from test_images/test.jpg...
Converting message: 'Hello World'
Loading encoder model from checkpoints/exp_20260210_235959/best_encoder.pth...
Encoding image...
Watermarked image saved to watermarked_images/test_watermarked.jpg
PSNR: 38.45 dB

✓ Encoding completed successfully!
```

#### 7.3 查看结果

打开 `watermarked_images/test_watermarked.jpg`，你会发现：
- 图片看起来和原图几乎一样
- 但是消息 "Hello World" 已经隐藏在里面了！

---

### 第八步：测试模型（解码图像）

#### 8.1 解码水印图像

在CMD中输入：

```cmd
python decode_image.py --image watermarked_images/test_watermarked.jpg --model checkpoints/exp_20260210_235959/best_decoder.pth
```

**你会看到：**
```
Loading watermarked image from watermarked_images/test_watermarked.jpg...
Loading decoder model from checkpoints/exp_20260210_235959/best_decoder.pth...
Decoding message...

Decoded message: 'Hello World'
Confidence: 0.9876

✓ Decoding completed successfully!
```

**成功！** 系统成功从图片中恢复了隐藏的消息！🎉

---

## 🎯 完整命令总结

### 1. 准备小数据集
```cmd
cd D:\大创 视觉\李威\李威\yolov5-6.2\invisible-watermark
python prepare_small_dataset.py
```

### 2. 开始训练
```cmd
python train.py --train-dir data/small/train --val-dir data/small/val --batch-size 16 --num-epochs 100 --tensorboard
```

### 3. 编码图像
```cmd
python encode_image.py --image test_images/test.jpg --message "Hello World" --model checkpoints/exp_XXXXXX_XXXXXX/best_encoder.pth --output watermarked_images/test_watermarked.jpg
```

### 4. 解码图像
```cmd
python decode_image.py --image watermarked_images/test_watermarked.jpg --model checkpoints/exp_XXXXXX_XXXXXX/best_decoder.pth
```

---

## ⚠️ 常见问题

### Q1: 训练中断了怎么办？

**答：** 模型会自动保存检查点。重新运行训练命令即可继续。

### Q2: 显存不足怎么办？

**答：** 减小批次大小：
```cmd
python train.py --train-dir data/small/train --val-dir data/small/val --batch-size 8 --num-epochs 100
```

### Q3: 想用更少的图片测试？

**答：** 修改数量参数：
```cmd
python prepare_small_dataset.py --num-train 100 --num-val 20
```

### Q4: 找不到模型文件？

**答：** 查看 checkpoints 文件夹，找到最新的 exp_ 开头的文件夹，使用里面的模型。

---

## 📊 预期效果

训练完成后，你的模型应该能够：

- ✅ 将32位消息隐藏到图片中
- ✅ 水印图片与原图视觉上几乎相同（PSNR > 35dB）
- ✅ 解码准确率 > 95%（BER < 0.05）
- ✅ 对JPEG压缩、裁剪等攻击有一定抵抗力

---

## 🎉 恭喜！

如果你完成了所有步骤，你已经成功：

1. ✅ 准备了训练数据集
2. ✅ 训练了隐形水印模型
3. ✅ 测试了编码和解码功能
4. ✅ 掌握了整个系统的使用方法

**下一步你可以：**
- 尝试不同的消息
- 测试更多图片
- 调整训练参数提高性能
- 使用更大的数据集训练更强的模型

---

**最后更新：** 2026年2月10日
**作者：** Kiro AI Assistant
