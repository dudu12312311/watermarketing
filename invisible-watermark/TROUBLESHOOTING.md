# 故障排除指南

## ✅ 已修复的问题

### 问题1：ImportError - CombinedLoss
**状态：** ✅ 已修复

### 问题2：KeyError - log_interval  
**状态：** ✅ 已修复

### 问题3：pin_memory 警告
**状态：** ✅ 已修复（改为False）

---

## 🧪 验证修复

运行测试脚本验证所有配置是否正常：

```cmd
python test_config.py
```

你应该看到：
```
✅ 配置文件测试通过！
```

---

## 🚀 现在可以开始训练了

### 方法1：使用默认配置（需要大数据集）

```cmd
python train.py --batch-size 32 --num-epochs 300 --tensorboard
```

### 方法2：使用小数据集（推荐新手）

```cmd
python train.py ^
    --train-dir data/small/train ^
    --val-dir data/small/val ^
    --batch-size 16 ^
    --num-epochs 100 ^
    --tensorboard
```

---

## 🔧 其他常见问题

### 问题1：找不到数据目录

**错误信息：**
```
Error loading data: [Errno 2] No such file or directory: 'data/train'
```

**解决方案：**
```cmd
# 创建数据目录
mkdir data\train
mkdir data\val

# 或使用小数据集
mkdir data\small\train
mkdir data\small\val
```

然后把图片放入这些文件夹。

---

### 问题2：CUDA out of memory（显存不足）

**错误信息：**
```
RuntimeError: CUDA out of memory
```

**解决方案：**
```cmd
# 方案1：减小批次大小
python train.py --batch-size 8

# 方案2：使用CPU训练
python train.py --device cpu

# 方案3：减小图像大小（编辑 config.py）
# 将 image_size 从 400 改为 256 或 128
```

---

### 问题3：没有安装 PyTorch

**错误信息：**
```
ModuleNotFoundError: No module named 'torch'
```

**解决方案：**
```cmd
# 如果有NVIDIA显卡
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 如果没有显卡
pip install torch torchvision torchaudio
```

---

### 问题4：没有安装其他依赖

**错误信息：**
```
ModuleNotFoundError: No module named 'PIL'
或
ModuleNotFoundError: No module named 'tqdm'
```

**解决方案：**
```cmd
pip install pillow numpy tqdm tensorboard
```

---

### 问题5：TensorBoard 无法启动

**错误信息：**
```
ModuleNotFoundError: No module named 'tensorboard'
```

**解决方案：**
```cmd
# 安装 TensorBoard
pip install tensorboard

# 或者不使用 TensorBoard
python train.py --batch-size 32 --num-epochs 300
# （去掉 --tensorboard 参数）
```

---

## 📝 检查清单

在运行训练前，确保：

- [ ] Python 3.6+ 已安装
- [ ] PyTorch 已安装（运行 `python -c "import torch; print(torch.__version__)"` 验证）
- [ ] 其他依赖已安装（运行 `python test_imports.py` 验证）
- [ ] 数据文件夹已创建并包含图片
- [ ] 有足够的磁盘空间（至少10GB）

---

## 🆘 获取帮助

如果遇到其他问题：

1. 运行 `python test_imports.py` 检查导入
2. 检查错误信息的具体内容
3. 查看本文档的"其他常见问题"部分
4. 查看 `NEXT_STEPS.md` 了解详细步骤

---

**最后更新：** 2026年2月10日
