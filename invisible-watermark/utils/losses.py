"""
损失函数 - 编码器和解码器的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLoss(nn.Module):
    """
    编码器损失函数
    
    L = λ1 * MSE(watermarked, original) + λ2 * BCE(decoded, message)
    """
    
    def __init__(self, mse_weight=1.0, bce_weight=1.0, use_l1=False):
        """
        初始化编码器损失
        
        Args:
            mse_weight: MSE损失的权重
            bce_weight: BCE损失的权重
            use_l1: 是否使用L1损失而不是MSE
        """
        super(EncoderLoss, self).__init__()
        self.mse_weight = mse_weight
        self.bce_weight = bce_weight
        self.use_l1 = use_l1
        
        if use_l1:
            self.image_loss = nn.L1Loss()
        else:
            self.image_loss = nn.MSELoss()
        
        self.message_loss = nn.BCELoss()
    
    def forward(self, watermarked_image, original_image, decoded_message, original_message):
        """
        计算编码器损失
        
        Args:
            watermarked_image: (B, 3, H, W) - 水印图像
            original_image: (B, 3, H, W) - 原始图像
            decoded_message: (B, message_length) - 解码的消息
            original_message: (B, message_length) - 原始消息
        
        Returns:
            loss: 总损失
            loss_dict: 损失字典
        """
        # 图像损失
        image_loss = self.image_loss(watermarked_image, original_image)
        
        # 消息损失
        message_loss = self.message_loss(decoded_message, original_message)
        
        # 总损失
        total_loss = self.mse_weight * image_loss + self.bce_weight * message_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'image': image_loss.item(),
            'message': message_loss.item(),
        }
        
        return total_loss, loss_dict


class DecoderLoss(nn.Module):
    """
    解码器损失函数
    
    L = BCE(decoded, message)
    """
    
    def __init__(self):
        """初始化解码器损失"""
        super(DecoderLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, decoded_message, original_message):
        """
        计算解码器损失
        
        Args:
            decoded_message: (B, message_length) - 解码的消息
            original_message: (B, message_length) - 原始消息
        
        Returns:
            loss: 损失值
        """
        loss = self.bce_loss(decoded_message, original_message)
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失函数 - 同时优化编码器和解码器
    
    L = L_encoder + λ * L_decoder
    """
    
    def __init__(self, encoder_mse_weight=1.0, encoder_bce_weight=1.0, 
                 decoder_weight=1.0, use_l1=False):
        """
        初始化组合损失
        
        Args:
            encoder_mse_weight: 编码器MSE损失的权重
            encoder_bce_weight: 编码器BCE损失的权重
            decoder_weight: 解码器损失的权重
            use_l1: 是否使用L1损失
        """
        super(CombinedLoss, self).__init__()
        self.encoder_loss = EncoderLoss(encoder_mse_weight, encoder_bce_weight, use_l1)
        self.decoder_loss = DecoderLoss()
        self.decoder_weight = decoder_weight
    
    def forward(self, watermarked_image, original_image, 
                decoded_message, original_message, 
                decoded_message_from_attacked=None):
        """
        计算组合损失
        
        Args:
            watermarked_image: (B, 3, H, W) - 水印图像
            original_image: (B, 3, H, W) - 原始图像
            decoded_message: (B, message_length) - 解码的消息
            original_message: (B, message_length) - 原始消息
            decoded_message_from_attacked: (B, message_length) - 从攻击图像解码的消息（可选）
        
        Returns:
            loss: 总损失
            loss_dict: 损失字典
        """
        # 编码器损失
        encoder_loss, encoder_loss_dict = self.encoder_loss(
            watermarked_image, original_image, decoded_message, original_message
        )
        
        # 解码器损失
        decoder_loss = self.decoder_loss(decoded_message, original_message)
        
        # 如果有攻击图像的解码结果，也计算其损失
        if decoded_message_from_attacked is not None:
            attacked_loss = self.decoder_loss(decoded_message_from_attacked, original_message)
            decoder_loss = (decoder_loss + attacked_loss) / 2
        
        # 总损失
        total_loss = encoder_loss + self.decoder_weight * decoder_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'encoder_total': encoder_loss_dict['total'],
            'encoder_image': encoder_loss_dict['image'],
            'encoder_message': encoder_loss_dict['message'],
            'decoder': decoder_loss.item(),
        }
        
        return total_loss, loss_dict


class PerceptualLoss(nn.Module):
    """
    感知损失 - 使用预训练的特征提取器
    """
    
    def __init__(self, feature_extractor=None, layer_weights=None):
        """
        初始化感知损失
        
        Args:
            feature_extractor: 特征提取器（如VGG）
            layer_weights: 各层的权重
        """
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        self.layer_weights = layer_weights or {}
    
    def forward(self, watermarked_image, original_image):
        """
        计算感知损失
        
        Args:
            watermarked_image: (B, 3, H, W) - 水印图像
            original_image: (B, 3, H, W) - 原始图像
        
        Returns:
            loss: 感知损失
        """
        if self.feature_extractor is None:
            # 如果没有特征提取器，使用L2损失
            return F.mse_loss(watermarked_image, original_image)
        
        # 提取特征
        features_watermarked = self.feature_extractor(watermarked_image)
        features_original = self.feature_extractor(original_image)
        
        # 计算特征损失
        loss = 0
        for i, (f_w, f_o) in enumerate(zip(features_watermarked, features_original)):
            weight = self.layer_weights.get(i, 1.0)
            loss += weight * F.mse_loss(f_w, f_o)
        
        return loss


if __name__ == '__main__':
    # 测试损失函数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    batch_size = 2
    message_length = 32
    
    watermarked = torch.randn(batch_size, 3, 100, 100).to(device)
    original = torch.randn(batch_size, 3, 100, 100).to(device)
    decoded_msg = torch.rand(batch_size, message_length).to(device)
    original_msg = torch.randint(0, 2, (batch_size, message_length)).float().to(device)
    
    # 测试编码器损失
    print("Testing EncoderLoss...")
    encoder_loss_fn = EncoderLoss(mse_weight=1.0, bce_weight=1.0)
    loss, loss_dict = encoder_loss_fn(watermarked, original, decoded_msg, original_msg)
    print(f"Encoder Loss: {loss.item():.6f}")
    print(f"Loss dict: {loss_dict}")
    
    # 测试解码器损失
    print("\nTesting DecoderLoss...")
    decoder_loss_fn = DecoderLoss()
    loss = decoder_loss_fn(decoded_msg, original_msg)
    print(f"Decoder Loss: {loss.item():.6f}")
    
    # 测试组合损失
    print("\nTesting CombinedLoss...")
    combined_loss_fn = CombinedLoss()
    loss, loss_dict = combined_loss_fn(watermarked, original, decoded_msg, original_msg)
    print(f"Combined Loss: {loss.item():.6f}")
    print(f"Loss dict: {loss_dict}")
