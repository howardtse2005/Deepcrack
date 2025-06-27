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
    """Authentic HiLo Attention from DECS-Net with crack-density-aware weighting"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, window_size=4, alpha=0.5, density_weight=1.0):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.ws = window_size
        self.alpha = alpha
        
        # Learnable density weighting parameter
        self.density_weight = nn.Parameter(torch.tensor(density_weight))
        self.eps = 1e-6  # Small constant to avoid division by zero
        
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
    
    def compute_edge_boost(self, features):
        """Compute crack-density-aware attention boosting for edge regions"""
        B, C, H, W = features.shape
        
        # Simple edge detector using gradient magnitude
        # This approximates crack density without requiring ground truth during inference
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=features.dtype, device=features.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=features.dtype, device=features.device)
        
        # Apply to mean across channels to get single gradient map
        feat_mean = features.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Compute gradients
        grad_x = F.conv2d(feat_mean, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        grad_y = F.conv2d(feat_mean, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        
        # Edge magnitude (proxy for crack edges)
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + self.eps)
        
        # Partition into windows for windowed attention
        if self.ws > 1:
            h_group, w_group = H // self.ws, W // self.ws
            edge_windowed = edge_magnitude.reshape(B, 1, h_group, self.ws, w_group, self.ws)
            edge_windowed = edge_windowed.permute(0, 2, 4, 3, 5, 1).reshape(B, h_group * w_group, self.ws * self.ws)
            
            # Compute edge density per window (higher values = more edges = need more attention)
            edge_density = edge_windowed.mean(dim=-1, keepdim=True)  # [B, total_groups, 1]
            
            # Boost attention for regions with more edges (crack boundaries)
            # Use learnable parameter to control the strength
            edge_boost = self.density_weight * torch.log(1 + edge_density)  # [B, total_groups, 1]
            
            return edge_boost.unsqueeze(-1)  # [B, total_groups, 1, 1] for broadcasting
        else:
            return None
    
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
            
            # Windowed self-attention with edge-aware boosting
            h_attn = (h_q @ h_k.transpose(-2, -1)) * self.scale  # [B, total_groups, h_heads, ws*ws, ws*ws]
            
            # Apply crack-density-aware edge boosting
            edge_boost = self.compute_edge_boost(high_feats)  # [B, total_groups, 1, 1]
            if edge_boost is not None:
                # Add edge boost to attention scores (more attention to edge regions)
                h_attn = h_attn + edge_boost.unsqueeze(-1).expand_as(h_attn)
            
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
    """Fusion between spatial and frequency streams with crack-density-aware cross-attention"""
    def __init__(self, dim, window_size=4, density_weight=0.5):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        
        # Learnable density weighting for fusion
        self.fusion_density_weight = nn.Parameter(torch.tensor(density_weight))
        self.eps = 1e-6
        
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
    
    def compute_fusion_edge_boost(self, spatial_feats, freq_feats):
        """Compute edge boosting for fusion based on spatial-frequency disagreement"""
        B, C, H, W = spatial_feats.shape
        
        # Compute feature difference to identify regions where spatial and frequency disagree
        # These are often edge/boundary regions that need more attention
        feat_diff = torch.abs(spatial_feats - freq_feats)
        disagreement = feat_diff.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Also use gradient-based edge detection on the disagreement map
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                              dtype=disagreement.dtype, device=disagreement.device)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                              dtype=disagreement.dtype, device=disagreement.device)
        
        # Compute gradients on disagreement map
        grad_x = F.conv2d(disagreement, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
        grad_y = F.conv2d(disagreement, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
        
        # Combined edge signal: disagreement + gradient magnitude
        edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + self.eps)
        combined_edge = disagreement + edge_magnitude
        
        # Partition into windows for windowed attention
        if self.ws > 1:
            h_group, w_group = H // self.ws, W // self.ws
            edge_windowed = combined_edge.reshape(B, 1, h_group, self.ws, w_group, self.ws)
            edge_windowed = edge_windowed.permute(0, 2, 4, 3, 5, 1).reshape(B, h_group * w_group, self.ws * self.ws)
            
            # Compute edge density per window
            edge_density = edge_windowed.mean(dim=-1, keepdim=True)  # [B, total_groups, 1]
            
            # Boost attention for high-disagreement regions (likely edges/boundaries)
            edge_boost = self.fusion_density_weight * torch.log(1 + edge_density)  # [B, total_groups, 1]
            
            return edge_boost.unsqueeze(-1)  # [B, total_groups, 1, 1] for broadcasting
        else:
            return None
        
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
        
        # Windowed cross-attention with disagreement-aware boosting
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, total_groups, num_heads, ws*ws, ws*ws]
        
        # Apply fusion edge boosting based on spatial-frequency disagreement
        fusion_edge_boost = self.compute_fusion_edge_boost(x_att, y_att)  # [B, total_groups, 1, 1]
        if fusion_edge_boost is not None:
            # Add edge boost to cross-attention scores (more attention to disagreement regions)
            attn = attn + fusion_edge_boost.unsqueeze(-1).expand_as(attn)
        
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
        
        # Single decoder path (shared)
        self.dec1 = Up(1024, 512)
        self.dec2 = Up(512, 256)
        self.dec3 = Up(256, 128)
        self.dec4 = Up(128, 64)
        
        # Output
        self.out_conv = nn.Conv2d(64, num_classes, 1)

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
        
        # Single decoder path using HiLo-enhanced skip connections where available
        d1 = self.dec1(f_bottleneck, t4_skip, s4_indices)  # Use HiLo-enhanced skip
        d2 = self.dec2(d1, t3_skip, s3_indices)           # Use HiLo-enhanced skip  
        d3 = self.dec3(d2, t2_skip, s2_indices)           # Original CNN features
        d4 = self.dec4(d3, t1_skip, s1_indices)           # Original CNN features
        
        # Output
        output = self.out_conv(d4)
        
        return output