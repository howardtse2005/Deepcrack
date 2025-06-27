import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from config import Config as cfg

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention for 2D feature maps with crack-density-aware weighting"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, dropout=0.1, density_weight=1.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Learnable density weighting parameter
        self.density_weight = nn.Parameter(torch.tensor(density_weight))
        self.eps = 1e-6  # Small constant to avoid division by zero
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def compute_edge_boost(self, x):
        """Compute crack-density-aware attention boosting for edge regions"""
        B, C, H, W = x.shape
        
        # Simple edge detector using gradient magnitude
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device)
        
        # Apply to mean across channels to get single gradient map
        feat_mean = x.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Compute gradients
        grad_x = F.conv2d(feat_mean, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        grad_y = F.conv2d(feat_mean, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        
        # Edge magnitude (proxy for crack edges)
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + self.eps)  # [B, 1, H, W]
        
        # Flatten to match attention dimensions
        edge_flat = edge_magnitude.flatten(2)  # [B, 1, H*W]
        edge_density = edge_flat.squeeze(1)  # [B, H*W]
        
        # Boost attention for regions with more edges (crack boundaries)
        edge_boost = self.density_weight * torch.log(1 + edge_density)  # [B, H*W]
        
        return edge_boost.unsqueeze(1).unsqueeze(-1)  # [B, 1, H*W, 1] for broadcasting
        
    def forward(self, x):
        B, C, H, W = x.shape
        # Reshape to sequence: [B, H*W, C]
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Generate Q, K, V
        qkv = self.qkv(x_flat).reshape(B, H*W, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, num_heads, H*W, head_dim]
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, H*W, H*W]
        
        # Apply crack-density-aware edge boosting
        edge_boost = self.compute_edge_boost(x)  # [B, 1, H*W, 1]
        if edge_boost is not None:
            # Add edge boost to attention scores (more attention to edge regions)
            attn = attn + edge_boost.expand_as(attn)
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H*W, C)  # [B, H*W, C]
        x_attn = self.proj(x_attn)
        x_attn = self.dropout(x_attn)
        
        # Reshape back to 2D: [B, C, H, W]
        x_attn = x_attn.transpose(1, 2).reshape(B, C, H, W)
        
        return x_attn

class TransformerBlock(nn.Module):
    """Transformer block with crack-density-aware self-attention and feed-forward"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0, qkv_bias=False, dropout=0.1, density_weight=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, qkv_bias, dropout, density_weight)
        self.norm2 = nn.LayerNorm(dim)
        
        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Self-attention with residual connection
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_norm1 = self.norm1(x_flat)
        x_norm1 = x_norm1.transpose(1, 2).reshape(B, C, H, W)
        
        attn_out = self.attn(x_norm1)
        x = x + attn_out  # Residual connection
        
        # MLP with residual connection
        x_flat = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        x_norm2 = self.norm2(x_flat)
        mlp_out = self.mlp(x_norm2)
        x_flat = x_flat + mlp_out  # Residual connection
        
        # Reshape back to 2D
        x = x_flat.transpose(1, 2).reshape(B, C, H, W)
        
        return x

def Conv3X3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=cfg.kernel_size, 
                     padding=cfg.kernel_size//2)

class DoubleConv(nn.Module):
    """Double convolution block with optional crack-density-aware transformer attention"""
    def __init__(self, in_channels, out_channels, use_attention=False, density_weight=1.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv3X3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv3X3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Add transformer attention if requested
        self.use_attention = use_attention
        if use_attention:
            self.attention = TransformerBlock(out_channels, num_heads=8, dropout=0.1, density_weight=density_weight)

    def forward(self, x):
        x = self.double_conv(x)
        if self.use_attention:
            x = self.attention(x)
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv with optional density-aware attention"""
    def __init__(self, in_channels, out_channels, use_attention=False, density_weight=1.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_attention=use_attention, density_weight=density_weight)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv with optional density-aware attention"""
    def __init__(self, in_channels, out_channels, bilinear=True, use_attention=False, density_weight=1.0):
        super().__init__()

        # If bilinear, use normal upsampling and convolution
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, use_attention=use_attention, density_weight=density_weight)
        else:
            # Use transpose convolution
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_attention=use_attention, density_weight=density_weight)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Adjust dimensions if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Final convolution layer"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """UNet with Crack-Density-Aware Transformer Attention at Bottleneck"""
    def __init__(self, num_classes=1, n_channels=3, bilinear=True, use_attention=True, density_weight=1.0):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.use_attention = use_attention

        # Channel sizes
        factor = 2 if bilinear else 1
        base_channels = 64
        
        # Initial convolution (no attention - too high resolution)
        self.inc = DoubleConv(n_channels, base_channels, use_attention=False)
        
        # Encoder with density-aware attention only at deepest layer
        self.down1 = Down(base_channels, base_channels*2, use_attention=False)
        self.down2 = Down(base_channels*2, base_channels*4, use_attention=False)
        self.down3 = Down(base_channels*4, base_channels*8, use_attention=False)
        self.down4 = Down(base_channels*8, base_channels*16 // factor, 
                         use_attention=use_attention, density_weight=density_weight)  # attention
        
        # Decoder with density-aware attention only at bottleneck level
        self.up1 = Up(base_channels*16, base_channels*8 // factor, bilinear, 
                      use_attention=use_attention, density_weight=density_weight)  # attention
        self.up2 = Up(base_channels*8, base_channels*4 // factor, bilinear, use_attention=False)
        self.up3 = Up(base_channels*4, base_channels*2 // factor, bilinear, use_attention=False)
        self.up4 = Up(base_channels*2, base_channels, bilinear, use_attention=False)
        
        # Output convolution
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with crack-density-aware attention at bottleneck
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Main output
        final_output = self.outc(x)
        
        return final_output
