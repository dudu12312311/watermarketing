"""
噪声层模块 - 模拟真实世界的图像处理攻击
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import re


class NoiseLayer(nn.Module):
    """噪声层基类"""
    
    def __init__(self):
        super(NoiseLayer, self).__init__()
    
    def forward(self, x):
        raise NotImplementedError


class Crop(NoiseLayer):
    """
    裁剪噪声层
    
    随机裁剪图像的指定比例
    """
    
    def __init__(self, height_range=(0.2, 0.25), width_range=(0.2, 0.25)):
        """
        Args:
            height_range: (min_ratio, max_ratio) - 保留高度的比例范围
            width_range: (min_ratio, max_ratio) - 保留宽度的比例范围
        """
        super(Crop, self).__init__()
        self.height_range = height_range
        self.width_range = width_range
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            cropped: (B, C, H, W) - 裁剪后的图像（填充到原始大小）
        """
        batch_size, channels, height, width = x.shape
        
        # 随机选择保留的比例
        height_ratio = random.uniform(self.height_range[0], self.height_range[1])
        width_ratio = random.uniform(self.width_range[0], self.width_range[1])
        
        # 计算裁剪后的大小
        crop_height = int(height * height_ratio)
        crop_width = int(width * width_ratio)
        
        # 随机选择裁剪位置
        top = random.randint(0, height - crop_height)
        left = random.randint(0, width - crop_width)
        
        # 裁剪
        cropped = x[:, :, top:top+crop_height, left:left+crop_width]
        
        # 上采样回原始大小
        cropped = F.interpolate(cropped, size=(height, width), mode='bilinear', align_corners=False)
        
        return cropped


class Cropout(NoiseLayer):
    """
    随机删除噪声层
    
    随机删除图像的指定区域
    """
    
    def __init__(self, height_range=(0.55, 0.6), width_range=(0.55, 0.6)):
        """
        Args:
            height_range: (min_ratio, max_ratio) - 删除区域高度的比例范围
            width_range: (min_ratio, max_ratio) - 删除区域宽度的比例范围
        """
        super(Cropout, self).__init__()
        self.height_range = height_range
        self.width_range = width_range
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            result: (B, C, H, W) - 删除后的图像
        """
        batch_size, channels, height, width = x.shape
        
        # 随机选择删除区域的大小
        delete_height = int(height * random.uniform(self.height_range[0], self.height_range[1]))
        delete_width = int(width * random.uniform(self.width_range[0], self.width_range[1]))
        
        # 随机选择删除位置
        top = random.randint(0, height - delete_height)
        left = random.randint(0, width - delete_width)
        
        # 创建结果
        result = x.clone()
        result[:, :, top:top+delete_height, left:left+delete_width] = 0
        
        return result


class Dropout(NoiseLayer):
    """
    像素丢弃噪声层
    
    随机丢弃像素
    """
    
    def __init__(self, keep_min=0.55, keep_max=0.6):
        """
        Args:
            keep_min: 保留像素的最小比例
            keep_max: 保留像素的最大比例
        """
        super(Dropout, self).__init__()
        self.keep_min = keep_min
        self.keep_max = keep_max
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            result: (B, C, H, W) - 丢弃后的图像
        """
        keep_ratio = random.uniform(self.keep_min, self.keep_max)
        
        # 创建掩码
        mask = torch.bernoulli(torch.full_like(x, keep_ratio))
        
        # 应用掩码
        result = x * mask
        
        return result


class Resize(NoiseLayer):
    """
    缩放噪声层
    
    随机缩放图像
    """
    
    def __init__(self, scale_min=0.7, scale_max=0.8):
        """
        Args:
            scale_min: 最小缩放比例
            scale_max: 最大缩放比例
        """
        super(Resize, self).__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            result: (B, C, H, W) - 缩放后的图像
        """
        batch_size, channels, height, width = x.shape
        
        # 随机选择缩放比例
        scale = random.uniform(self.scale_min, self.scale_max)
        
        # 计算新的大小
        new_height = int(height * scale)
        new_width = int(width * scale)
        
        # 缩放
        resized = F.interpolate(x, size=(new_height, new_width), mode='bilinear', align_corners=False)
        
        # 上采样回原始大小
        result = F.interpolate(resized, size=(height, width), mode='bilinear', align_corners=False)
        
        return result


class JPEG(NoiseLayer):
    """
    JPEG压缩噪声层
    
    可微分的JPEG压缩近似
    """
    
    def __init__(self, quality=75):
        """
        Args:
            quality: JPEG质量（1-100）
        """
        super(JPEG, self).__init__()
        self.quality = quality
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        
        Returns:
            result: (B, C, H, W) - 压缩后的图像
        """
        # 简化的JPEG压缩模拟：添加高频噪声
        # 实际的可微分JPEG实现会更复杂
        
        batch_size, channels, height, width = x.shape
        
        # 计算压缩强度（质量越低，压缩越强）
        compression_strength = (100 - self.quality) / 100.0 * 0.1
        
        # 添加高频噪声
        noise = torch.randn_like(x) * compression_strength
        result = x + noise
        
        # 限制范围
        result = torch.clamp(result, -1, 1)
        
        return result


class NoiseLayerContainer(nn.Module):
    """
    噪声层容器 - 管理多个噪声层的组合
    """
    
    def __init__(self, noise_config=None, use_random=True):
        """
        Args:
            noise_config: 噪声层配置字符串列表或单个字符串
            use_random: 是否随机选择噪声层（True）或顺序应用（False）
        """
        super(NoiseLayerContainer, self).__init__()
        
        self.use_random = use_random
        self.noise_layers = nn.ModuleList()
        
        if noise_config is None:
            noise_config = []
        
        # 处理配置
        if isinstance(noise_config, str):
            # 解析单个配置字符串
            self.noise_layers.append(self._parse_noise_layer(noise_config))
        elif isinstance(noise_config, list):
            # 处理配置列表
            for config in noise_config:
                self.noise_layers.append(self._parse_noise_layer(config))
    
    def _parse_noise_layer(self, config_str):
        """
        解析噪声层配置字符串
        
        支持的格式：
        - 'crop((0.2,0.25),(0.2,0.25))'
        - 'cropout((0.55,0.6),(0.55,0.6))'
        - 'dropout(0.55,0.6)'
        - 'resize(0.7,0.8)'
        - 'jpeg()'
        - 'crop(...)+cropout(...)+...'
        """
        config_str = config_str.strip()
        
        # 检查是否是组合配置
        if '+' in config_str:
            layers = []
            for part in config_str.split('+'):
                layers.append(self._parse_single_noise_layer(part.strip()))
            return nn.Sequential(*layers)
        else:
            return self._parse_single_noise_layer(config_str)
    
    def _parse_single_noise_layer(self, config_str):
        """解析单个噪声层"""
        config_str = config_str.strip()
        
        # 提取层类型和参数
        match = re.match(r'(\w+)\((.*)\)', config_str)
        if not match:
            raise ValueError(f"Invalid noise layer config: {config_str}")
        
        layer_type = match.group(1).lower()
        params_str = match.group(2)
        
        if layer_type == 'crop':
            # 解析 crop((h_min,h_max),(w_min,w_max))
            params = re.findall(r'\(([\d.,]+)\)', params_str)
            if len(params) == 2:
                h_range = tuple(map(float, params[0].split(',')))
                w_range = tuple(map(float, params[1].split(',')))
                return Crop(h_range, w_range)
            else:
                # 简化格式：crop(0.2,0.25)
                values = list(map(float, params_str.split(',')))
                if len(values) == 2:
                    return Crop((values[0], values[1]), (values[0], values[1]))
        
        elif layer_type == 'cropout':
            params = re.findall(r'\(([\d.,]+)\)', params_str)
            if len(params) == 2:
                h_range = tuple(map(float, params[0].split(',')))
                w_range = tuple(map(float, params[1].split(',')))
                return Cropout(h_range, w_range)
            else:
                values = list(map(float, params_str.split(',')))
                if len(values) == 2:
                    return Cropout((values[0], values[1]), (values[0], values[1]))
        
        elif layer_type == 'dropout':
            values = list(map(float, params_str.split(',')))
            if len(values) == 2:
                return Dropout(values[0], values[1])
        
        elif layer_type == 'resize':
            values = list(map(float, params_str.split(',')))
            if len(values) == 2:
                return Resize(values[0], values[1])
        
        elif layer_type == 'jpeg':
            if params_str.strip():
                quality = int(params_str)
                return JPEG(quality)
            else:
                return JPEG()
        
        raise ValueError(f"Unknown noise layer type: {layer_type}")
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: (B, C, H, W)
        
        Returns:
            result: (B, C, H, W)
        """
        if not self.noise_layers:
            return x
        
        if self.use_random:
            # 随机选择一个噪声层
            noise_layer = random.choice(self.noise_layers)
            return noise_layer(x)
        else:
            # 顺序应用所有噪声层
            result = x
            for noise_layer in self.noise_layers:
                result = noise_layer(result)
            return result


if __name__ == '__main__':
    # 测试噪声层
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试图像
    image = torch.randn(2, 3, 400, 400).to(device)
    
    # 测试各个噪声层
    print("Testing noise layers...")
    
    crop = Crop((0.2, 0.25), (0.2, 0.25)).to(device)
    cropped = crop(image)
    print(f"Crop: {image.shape} -> {cropped.shape}")
    
    cropout = Cropout((0.55, 0.6), (0.55, 0.6)).to(device)
    cropout_result = cropout(image)
    print(f"Cropout: {image.shape} -> {cropout_result.shape}")
    
    dropout = Dropout(0.55, 0.6).to(device)
    dropout_result = dropout(image)
    print(f"Dropout: {image.shape} -> {dropout_result.shape}")
    
    resize = Resize(0.7, 0.8).to(device)
    resized = resize(image)
    print(f"Resize: {image.shape} -> {resized.shape}")
    
    jpeg = JPEG(75).to(device)
    jpeg_result = jpeg(image)
    print(f"JPEG: {image.shape} -> {jpeg_result.shape}")
    
    # 测试容器
    print("\nTesting noise layer container...")
    container = NoiseLayerContainer(
        'crop((0.2,0.25),(0.2,0.25))+jpeg()',
        use_random=False
    ).to(device)
    result = container(image)
    print(f"Combined noise: {image.shape} -> {result.shape}")
