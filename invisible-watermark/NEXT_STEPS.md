# HiDDeN 隐形水印系统 - 后续工作指南

## 📋 项目现状

✅ **代码实现完成** - 所有核心模块已实现
✅ **文档完整** - 详细的使用指南和API文档
✅ **测试框架就绪** - 完整的测试和评估工具

🔄 **下一步** - 数据准备、模型训练、水印生成

---

## 🎯 Phase 11: 数据准备与模型训练

### 步骤 1: 准备COCO数据集

#### 1.1 下载数据集

访问 [COCO官网](http://cocodataset.org/#download) 下载2017版本：

```bash
# 创建数据目录
mkdir -p data/train data/val

# 下载COCO 2017训练集（约18GB）
# 下载COCO 2017验证集（约1GB）
# 从官网下载或使用以下脚本
```

#### 1.2 组织数据结构

```
data/
├── train/          # 10,000张训练图像
│   ├── 000000000001.jpg
│   ├── 000000000002.jpg
│   └── ...
└── val/            # 1,000张验证图像
    ├── 000000000001.jpg
    ├── 000000000002.jpg
    └── ...
```

#### 1.3 验证数据

```bash
# 检查训练集
ls data/train | wc -l  # 应该显示 10000

# 检查验证集
ls data/val | wc -l    # 应该显示 1000
```

---

### 步骤 2: 训练基础模型（无噪声）

这是最基础的训练，用于验证系统是否正常工作。

```bash
# 基础训练（无噪声层）
python train.py \
    --batch-size 32 \
    --num-epochs 300 \
    --learning-rate 1e-4 \
    --tensorboard

# 启用TensorBoard可视化
tensorboard --logdir logs
```

**预期结果**：
- Encoder MSE: < 0.01
- Bitwise Error: < 0.001
- PSNR: > 40dB

**输出**：
- 模型检查点保存在 `checkpoints/exp_YYYYMMDD_HHMMSS/`
- TensorBoard日志保存在 `logs/`

---

### 步骤 3: 训练单个噪声层模型

分别训练5种噪声层配置，每种都能提高模型的鲁棒性。

#### 3.1 Crop噪声（裁剪）

```bash
python train.py \
    --batch-size 32 \
    --num-epochs 300 \
    --noise 'crop((0.2,0.25),(0.2,0.25))' \
    --tensorboard
```

**说明**：
- 随机裁剪图像到原始大小的 ~5%
- 模拟图像裁剪攻击

**预期结果**：
- Encoder MSE: 0.046
- Bitwise Error: 0.0019
- Decoder MSE: 0.0435

#### 3.2 Cropout噪声（随机删除）

```bash
python train.py \
    --batch-size 32 \
    --num-epochs 300 \
    --noise 'cropout((0.55,0.6),(0.55,0.6))' \
    --tensorboard
```

**说明**：
- 随机删除图像的 ~33% 区域
- 模拟图像部分丢失

**预期结果**：
- Encoder MSE: 0.071
- Bitwise Error: 0.0011
- Decoder MSE: 0.0662

#### 3.3 Dropout噪声（像素丢弃）

```bash
python train.py \
    --batch-size 32 \
    --num-epochs 300 \
    --noise 'dropout(0.55,0.6)' \
    --tensorboard
```

**说明**：
- 随机丢弃 ~45% 的像素
- 模拟图像噪声

**预期结果**：
- Encoder MSE: 0.033
- Bitwise Error: 0.0019
- Decoder MSE: 0.0298

#### 3.4 Resize噪声（缩放）

```bash
python train.py \
    --batch-size 32 \
    --num-epochs 300 \
    --noise 'resize(0.7,0.8)' \
    --tensorboard
```

**说明**：
- 随机缩放图像到 70%-80%
- 模拟图像缩放

**预期结果**：
- Encoder MSE: 0.0251
- Bitwise Error: 0.0016
- Decoder MSE: 0.0238

#### 3.5 JPEG噪声（压缩）

```bash
python train.py \
    --batch-size 32 \
    --num-epochs 300 \
    --noise 'jpeg()' \
    --tensorboard
```

**说明**：
- 应用可微分的JPEG压缩
- 模拟JPEG压缩攻击

**预期结果**：
- Encoder MSE: 0.0272
- Bitwise Error: 0.0025
- Decoder MSE: 0.0253

---

### 步骤 4: 训练组合噪声模型

这是最复杂的配置，结合多种噪声层。

```bash
python train.py \
    --batch-size 32 \
    --num-epochs 400 \
    --noise 'crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()' \
    --tensorboard
```

**说明**：
- 随机选择一个噪声层应用到每个批次
- 提供最高的鲁棒性

**预期结果**：
- Encoder MSE: 0.1681
- Bitwise Error: 0.0028
- Decoder MSE: 0.1648

---

## 🎨 Phase 12: 生成水印图像

### 步骤 1: 准备测试图像

```bash
# 创建测试目录
mkdir -p test_images watermarked_images

# 放入你的测试图像（JPG/PNG格式）
# 建议至少10张不同的图像
```

### 步骤 2: 编码单张图像

```bash
python encode_image.py \
    --image test_images/sample.jpg \
    --message "Hello World" \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_encoder.pth \
    --output watermarked_images/sample_watermarked.jpg
```

**输出**：
- 水印图像保存到指定路径
- 控制台输出PSNR值

### 步骤 3: 批量编码

```bash
# Linux/Mac
for img in test_images/*.jpg; do
    filename=$(basename "$img")
    python encode_image.py \
        --image "$img" \
        --message "Secret Message" \
        --model checkpoints/exp_YYYYMMDD_HHMMSS/best_encoder.pth \
        --output "watermarked_images/${filename%.*}_watermarked.jpg"
done

# Windows (PowerShell)
Get-ChildItem test_images/*.jpg | ForEach-Object {
    python encode_image.py `
        --image $_.FullName `
        --message "Secret Message" `
        --model checkpoints/exp_YYYYMMDD_HHMMSS/best_encoder.pth `
        --output "watermarked_images/$($_.BaseName)_watermarked.jpg"
}
```

### 步骤 4: 解码水印图像

```bash
# 单张图像解码
python decode_image.py \
    --image watermarked_images/sample_watermarked.jpg \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_decoder.pth

# 批量解码
for img in watermarked_images/*.jpg; do
    echo "Decoding: $img"
    python decode_image.py \
        --image "$img" \
        --model checkpoints/exp_YYYYMMDD_HHMMSS/best_decoder.pth
done
```

### 步骤 5: 测试鲁棒性

```bash
# 测试Crop攻击
python test.py \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_model.pth \
    --test-images watermarked_images/ \
    --noise 'crop((0.2,0.25),(0.2,0.25))'

# 测试JPEG压缩
python test.py \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_model.pth \
    --test-images watermarked_images/ \
    --noise 'jpeg()'

# 测试组合噪声
python test.py \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_model.pth \
    --test-images watermarked_images/ \
    --noise 'crop((0.2,0.25),(0.2,0.25))+jpeg()'
```

---

## 📊 Phase 13: 性能评估

### 步骤 1: 运行完整测试

```bash
python test.py \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_model.pth \
    --test-images watermarked_images/ \
    --output test_results.json
```

### 步骤 2: 分析结果

```bash
# 查看测试结果
cat test_results.json | python -m json.tool
```

### 步骤 3: 生成报告

测试脚本会自动生成：
- `test_results.json` - 详细的测试数据
- `test_report.txt` - 可读的测试报告
- 性能可视化图表

---

## 🔧 常见问题与解决方案

### Q1: 显存不足

**症状**：`RuntimeError: CUDA out of memory`

**解决方案**：
```bash
# 减小批大小
python train.py --batch-size 16

# 或使用CPU训练
python train.py --device cpu
```

### Q2: 训练速度慢

**症状**：每个epoch耗时很长

**解决方案**：
```bash
# 增加数据加载线程
python train.py --num-workers 8

# 减小图像大小
# 编辑 config.py 中的 IMAGE_SIZE
```

### Q3: 解码准确率低

**症状**：Bitwise Error > 0.05

**解决方案**：
- 增加训练轮数：`--num-epochs 400`
- 调整学习率：`--learning-rate 5e-5`
- 检查数据质量

### Q4: 模型检查点路径

**症状**：找不到模型文件

**解决方案**：
```bash
# 查看可用的检查点
ls -la checkpoints/

# 使用完整路径
python encode_image.py \
    --image test.jpg \
    --model ./checkpoints/exp_20260210_120000/best_encoder.pth \
    --output output.jpg
```

---

## 📈 性能对标

### 参考实现结果（来自GitHub）

| 配置 | Encoder MSE | Bitwise Error | Decoder MSE | Epochs |
|------|------------|---------------|------------|--------|
| 无噪声 | < 0.01 | < 0.001 | < 0.01 | 300 |
| Crop | 0.046 | 0.0019 | 0.0435 | 300 |
| Cropout | 0.071 | 0.0011 | 0.0662 | 300 |
| Dropout | 0.033 | 0.0019 | 0.0298 | 300 |
| JPEG | 0.0272 | 0.0025 | 0.0253 | 300 |
| Resize | 0.0251 | 0.0016 | 0.0238 | 300 |
| 组合噪声 | 0.1681 | 0.0028 | 0.2109 | 400 |

---

## 📝 建议的工作流程

### 第1天：数据准备
1. 下载COCO数据集
2. 组织数据结构
3. 验证数据完整性

### 第2-3天：基础训练
1. 训练无噪声模型
2. 验证基础功能
3. 调整超参数

### 第4-5天：单噪声训练
1. 训练5种单噪声模型
2. 对比性能
3. 选择最佳配置

### 第6天：组合噪声训练
1. 训练组合噪声模型
2. 评估鲁棒性
3. 生成最终模型

### 第7天：水印生成与测试
1. 生成水印图像
2. 测试鲁棒性
3. 生成性能报告

---

## 🎓 学习资源

- **原始论文**：[HiDDeN: Hiding Data With Deep Networks](https://arxiv.org/abs/1807.09937)
- **参考实现**：[GitHub - dudu12312311/watermarketing](https://github.com/dudu12312311/watermarketing)
- **COCO数据集**：[http://cocodataset.org/](http://cocodataset.org/)

---

## ✅ 检查清单

在开始训练前，请确保：

- [ ] Python 3.6+ 已安装
- [ ] PyTorch 1.0+ 已安装
- [ ] 所有依赖已安装：`pip install -r requirements.txt`
- [ ] COCO数据集已下载并组织好
- [ ] 有足够的磁盘空间（至少50GB用于数据和模型）
- [ ] GPU可用（可选，但推荐）
- [ ] TensorBoard已安装（可选）

---

## 🚀 快速开始命令

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 训练基础模型
python train.py --batch-size 32 --num-epochs 300 --tensorboard

# 3. 编码图像
python encode_image.py \
    --image test.jpg \
    --message "Hello" \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_encoder.pth \
    --output watermarked.jpg

# 4. 解码图像
python decode_image.py \
    --image watermarked.jpg \
    --model checkpoints/exp_YYYYMMDD_HHMMSS/best_decoder.pth

# 5. 运行测试
python test.py --model checkpoints/exp_YYYYMMDD_HHMMSS/best_model.pth
```

---

**最后更新**：2026年2月10日
**项目状态**：✅ 代码完成，准备训练
