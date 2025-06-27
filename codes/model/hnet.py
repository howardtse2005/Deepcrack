import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from config import Config as cfg

try:
    from pytorch_wavelets import DWTForward, DWTInverse
    WAVELETS_AVAILABLE = True
except ImportError:
    print("Warning: pytorch_wavelets not available. Install with: pip install pytorch_wavelets")
    WAVELETS_AVAILABLE = False

def Conv3X3(in_, out):
    return nn.Conv2d(in_, out, cfg.kernel_size, padding=cfg.kernel_size//2)

class Down(nn.Module):
    """U-Net encoder block with index tracking for precise unpooling"""
    def __init__(self, in_ch, out_ch, downsample=True):
        super().__init__()
        self.conv = nn.Sequential(
            Conv3X3(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            Conv3X3(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)

    def forward(self, x):
        x = self.conv(x)
        if self.downsample:
            output, indices = self.pool(x)
            return output, x, indices
        return x

class Up(nn.Module):
    """U-Net decoder block with precise unpooling and dimension adaptation"""
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.channel_adapter = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        self.unpool = nn.MaxUnpool2d(2, stride=2)
        self.conv = nn.Sequential(
            Conv3X3(out_ch*2, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            Conv3X3(out_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x, skip, indices):
        x = self.channel_adapter(x)
        x = self.unpool(x, indices)
        if x.shape[2:] != skip.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class DSC(nn.Module):
    """Depthwise Separable Convolution"""
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(DSC, self).__init__()
        self.dw = nn.Conv2d(c_in, c_in, k_size, stride, padding, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out

class IDSC(nn.Module):
    """Inverted Depthwise Separable Convolution"""
    def __init__(self, c_in, c_out, k_size=3, stride=1, padding=1):
        super(IDSC, self).__init__()
        self.pw = nn.Conv2d(c_in, c_out, 1, 1)
        self.dw = nn.Conv2d(c_out, c_out, k_size, stride, padding, groups=c_out)

    def forward(self, x):
        out = self.pw(x)
        out = self.dw(out)
        return out

class HiLoAttention(nn.Module):
    """Authentic HiLo Attention from DECS-Net with self-attention"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, window_size=4, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ws = window_size
        self.alpha = alpha
        
        # Split heads between low and high frequency
        self.l_heads = int(num_heads * alpha)
        self.l_dim = self.l_heads * self.head_dim
        self.h_heads = num_heads - self.l_heads  
        self.h_dim = self.h_heads * self.head_dim
        
        # Handle edge case where all heads go to low frequency
        if self.ws == 1:
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim
        
        # Low-frequency decomposition via Haar wavelet transform
        if self.ws != 1 and WAVELETS_AVAILABLE:
            # Use actual Haar DWT
            self.dwt = DWTForward(J=1, mode='zero', wave='haar')
            self.idwt = DWTInverse(mode='zero', wave='haar')
            self.restore = nn.Conv2d(dim*3, dim, 1)  # Process 3 high-freq subbands (LH, HL, HH)
        elif self.ws != 1:
            # Fallback to average pooling if pytorch_wavelets not available
            self.avg_pool = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.sr_up = nn.PixelShuffle(window_size)
            self.restore = nn.Conv2d(dim//(window_size*window_size), dim, 1)
        
        # Low-frequency attention (global)
        if self.l_heads > 0:
            self.l_q = DSC(dim, self.l_dim)
            self.l_kv = DSC(dim, self.l_dim * 2) 
            self.l_proj = DSC(self.l_dim, self.l_dim)
        
        # High-frequency attention (windowed)
        if self.h_heads > 0:
            self.h_qkv = DSC(dim, self.h_dim * 3)
            self.h_proj = DSC(self.h_dim, self.h_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Low-frequency decomposition
        if self.ws != 1:
            if WAVELETS_AVAILABLE:
                # Use Haar DWT for proper frequency decomposition
                low_feats, high_feats_list = self.dwt(x)  # low_feats: [B, C, H//2, W//2], high_feats_list: list of [B, C, H//2, W//2]
                # Extract high-frequency components (LH, HL, HH)
                high_feats_dwt = high_feats_list[0]  # [B, C, 3, H//2, W//2] 
                high_feats_dwt = high_feats_dwt.view(B, C*3, H//2, W//2)  # Flatten LH, HL, HH subbands
                high_feats = self.restore(high_feats_dwt)  # Process high-freq components
                # Upsample to original resolution
                high_feats = F.interpolate(high_feats, size=(H, W), mode='bilinear', align_corners=False)
                # Get residual (true high-frequency content)
                low_upsampled = F.interpolate(low_feats, size=(H, W), mode='bilinear', align_corners=False)
                high_feats = x - low_upsampled  # True high-frequency residual
            else:
                # Fallback to average pooling
                low_feats = self.avg_pool(x)  # [B, C, H//ws, W//ws]
                low_up = self.sr_up(low_feats)  # [B, C//(ws*ws), H, W]  
                high_feats = self.restore(low_up) - x  # High-freq residual
        else:
            low_feats = x
            high_feats = x
        
        outputs = []
        
        # Low-frequency self-attention (global)
        if self.l_heads > 0:
            # Query from original, Key/Value from low-frequency
            l_q = self.l_q(x).reshape(B, self.l_heads, self.head_dim, H * W).transpose(-2, -1)  # [B, l_heads, HW, head_dim]
            
            if self.ws > 1:
                l_h, l_w = low_feats.shape[2:]
                l_kv = self.l_kv(low_feats).reshape(B, 2, self.l_heads, self.head_dim, l_h * l_w).permute(1, 0, 2, 4, 3)  # [2, B, l_heads, l_hw, head_dim]
            else:
                l_kv = self.l_kv(x).reshape(B, 2, self.l_heads, self.head_dim, H * W).permute(1, 0, 2, 4, 3)
            
            l_k, l_v = l_kv[0], l_kv[1]  # [B, l_heads, l_hw, head_dim]
            
            # Self-attention: Q @ K^T
            l_attn = (l_q @ l_k.transpose(-2, -1)) * self.scale  # [B, l_heads, HW, l_hw]
            l_attn = l_attn.softmax(dim=-1)
            
            # Apply attention to values
            l_x = (l_attn @ l_v).transpose(-2, -1).reshape(B, self.l_dim, H, W)  # [B, l_dim, H, W]
            l_x = self.l_proj(l_x)
            outputs.append(l_x)
        
        # High-frequency windowed self-attention  
        if self.h_heads > 0:
            # Partition into windows
            h_group, w_group = H // self.ws, W // self.ws
            total_groups = h_group * w_group
            
            # Reshape into windows: [B, h_group, w_group, ws, ws, C] -> [B, total_groups, ws*ws, C]
            h_feats_windowed = high_feats.reshape(B, C, h_group, self.ws, w_group, self.ws)
            h_feats_windowed = h_feats_windowed.permute(0, 2, 4, 3, 5, 1).reshape(B, total_groups, self.ws * self.ws, C)
            
            # Generate Q, K, V for each window
            h_qkv = self.h_qkv(high_feats).reshape(B, 3 * self.h_heads, self.head_dim, h_group, self.ws, w_group, self.ws)
            h_qkv = h_qkv.permute(0, 3, 5, 1, 4, 6, 2).reshape(B, total_groups, 3, self.h_heads, self.ws * self.ws, self.head_dim)
            h_qkv = h_qkv.permute(2, 0, 1, 3, 4, 5)  # [3, B, total_groups, h_heads, ws*ws, head_dim]
            
            h_q, h_k, h_v = h_qkv[0], h_qkv[1], h_qkv[2]
            
            # Windowed self-attention
            h_attn = (h_q @ h_k.transpose(-2, -1)) * self.scale  # [B, total_groups, h_heads, ws*ws, ws*ws]
            h_attn = h_attn.softmax(dim=-1)
            
            # Apply attention
            h_attn_out = (h_attn @ h_v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
            h_x = h_attn_out.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)
            h_x = h_x.permute(0, 3, 1, 2)  # [B, h_dim, H, W]
            h_x = self.h_proj(h_x)
            outputs.append(h_x)
        
        # Combine low and high frequency outputs
        if len(outputs) == 2:
            out = torch.cat(outputs, dim=1)  # [B, l_dim + h_dim, H, W]
        elif self.l_heads == 0:
            out = outputs[0]  # Only high-freq
        else:
            out = outputs[0]  # Only low-freq
            
        return out

class Fusion(nn.Module):
    """Fusion between spatial and frequency streams with windowed cross-attention"""
    def __init__(self, dim, window_size=4):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        
        # Channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, max(dim // 8, 1), 1),
            nn.ReLU(),
            nn.Conv2d(max(dim // 8, 1), dim * 2, 1),
            nn.Sigmoid()
        )
        
        # Windowed cross-attention (like original DECS-Net)
        self.num_heads = 8
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_conv = DSC(dim, dim)
        self.k_conv = DSC(dim, dim)  
        self.v_conv = DSC(dim, dim)
        self.proj = DSC(dim, dim)
        
        # Final fusion
        self.final_conv = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),  # [x, y, cross_attn]
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        
    def forward(self, x, y):
        B, C, H, W = x.shape
        
        # Element-wise multiplication
        element_wise = x * y
        
        # Channel attention
        concat = torch.cat([x, y], dim=1)
        ca_weights = self.channel_attn(concat)
        attended = concat * ca_weights
        
        # Split back
        x_att, y_att = torch.chunk(attended, 2, dim=1)
        
        # Windowed cross-attention (y queries x)
        h_group, w_group = H // self.ws, W // self.ws
        total_groups = h_group * w_group
        
        # Generate Q, K, V
        q = self.q_conv(y_att)  # Query from transformer features
        k = self.k_conv(x_att)  # Key from CNN features
        v = self.v_conv(x_att)  # Value from CNN features
        
        # Reshape into windows for windowed attention
        q = q.reshape(B, self.num_heads, self.head_dim, h_group, self.ws, w_group, self.ws)
        q = q.permute(0, 3, 5, 1, 4, 6, 2).reshape(B, total_groups, self.num_heads, self.ws * self.ws, self.head_dim)
        
        k = k.reshape(B, self.num_heads, self.head_dim, h_group, self.ws, w_group, self.ws)
        k = k.permute(0, 3, 5, 1, 4, 6, 2).reshape(B, total_groups, self.num_heads, self.ws * self.ws, self.head_dim)
        
        v = v.reshape(B, self.num_heads, self.head_dim, h_group, self.ws, w_group, self.ws)
        v = v.permute(0, 3, 5, 1, 4, 6, 2).reshape(B, total_groups, self.num_heads, self.ws * self.ws, self.head_dim)
        
        # Windowed cross-attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, total_groups, num_heads, ws*ws, ws*ws]
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        cross_out = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, C)
        cross_out = cross_out.transpose(2, 3).reshape(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]
        cross_out = self.proj(cross_out)
        
        # Combine all features
        fused = torch.cat([element_wise, y_att, cross_out], dim=1)
        output = self.final_conv(fused)
        
        return output

class HNet(nn.Module):
    """Memory-efficient HNet with simplified DECS-Net components"""
    def __init__(self, num_classes=1):
        super().__init__()
        
        # Shared encoder for both streams (memory efficient)
        self.enc1 = Down(3, 64)
        self.enc2 = Down(64, 128) 
        self.enc3 = Down(128, 256)
        self.enc4 = Down(256, 512)
        
        # HiLo attention blocks (only at higher levels to save memory)
        self.hilo3 = HiLoAttention(256, num_heads=8, window_size=4, alpha=0.2)
        self.hilo4 = HiLoAttention(512, num_heads=8, window_size=4, alpha=0.1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            Conv3X3(512, 1024),
            nn.ReLU()
        )
        
        # Fusion module (only at bottleneck for memory efficiency)
        self.fusion_bottleneck = Fusion(1024)
        
        # Dual decoder paths (maintaining HNet architecture)
        self.spatial_dec1 = Up(1024, 512)
        self.spatial_dec2 = Up(512, 256)
        self.spatial_dec3 = Up(256, 128)
        self.spatial_dec4 = Up(128, 64)
        self.spatial_out = nn.Conv2d(64, num_classes, 1)
        
        self.freq_dec1 = Up(1024, 512)
        self.freq_dec2 = Up(512, 256)
        self.freq_dec3 = Up(256, 128)
        self.freq_dec4 = Up(128, 64)
        self.freq_out = nn.Conv2d(64, num_classes, 1)
        
        # Final combination
        self.final_conv = nn.Sequential(
            Conv3X3(2, 32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        # Encoder path 1 (Spatial - normal CNN)
        s1, s1_skip, s1_indices = self.enc1(x)
        s2, s2_skip, s2_indices = self.enc2(s1)
        s3, s3_skip, s3_indices = self.enc3(s2)
        s4, s4_skip, s4_indices = self.enc4(s3)
        s_bottleneck = self.bottleneck(s4)
        
        # Encoder path 2 (Frequency - with HiLo attention at levels 3&4)
        t1, t1_skip, t1_indices = self.enc1(x)  # Reuse encoder
        t2, t2_skip, t2_indices = self.enc2(t1)
        
        # Apply HiLo attention at level 3 and 4 only
        t3, t3_skip, t3_indices = self.enc3(t2)
        t3_skip = self.hilo3(t3_skip)  # HiLo attention at level 3
        
        t4, t4_skip, t4_indices = self.enc4(t3)
        t4_skip = self.hilo4(t4_skip)  # HiLo attention at level 4
        
        t_bottleneck = self.bottleneck(t4)
        
        # Fusion (only at bottleneck for memory efficiency)
        f_bottleneck = self.fusion_bottleneck(s_bottleneck, t_bottleneck)
        
        # Spatial decoder path (uses original CNN skip connections)
        sd1 = self.spatial_dec1(f_bottleneck, s4_skip, s4_indices)
        sd2 = self.spatial_dec2(sd1, s3_skip, s3_indices)
        sd3 = self.spatial_dec3(sd2, s2_skip, s2_indices)
        sd4 = self.spatial_dec4(sd3, s1_skip, s1_indices)
        
        # Frequency decoder path (uses HiLo-enhanced skip connections where available)
        fd1 = self.freq_dec1(f_bottleneck, t4_skip, t4_indices)  # HiLo-enhanced
        fd2 = self.freq_dec2(fd1, t3_skip, t3_indices)           # HiLo-enhanced
        fd3 = self.freq_dec3(fd2, t2_skip, t2_indices)           # Original CNN features
        fd4 = self.freq_dec4(fd3, t1_skip, t1_indices)           # Original CNN features
        
        # Generate decoder outputs
        spatial_out = self.spatial_out(sd4)
        freq_out = self.freq_out(fd4)
        
        # Combine spatial and frequency decoder outputs
        spatial_prob = torch.sigmoid(spatial_out)
        freq_prob = torch.sigmoid(freq_out)
        combined = torch.cat([spatial_prob, freq_prob], dim=1)
        final_output = self.final_conv(combined)
        
        return final_output