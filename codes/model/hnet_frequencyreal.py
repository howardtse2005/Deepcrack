import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from config import Config as cfg

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

class FFTProcessor(nn.Module):
    """FFT-based frequency domain processing"""
    def __init__(self):
        super().__init__()
        # Learnable frequency filtering weights
        self.freq_weights = nn.Parameter(torch.ones(1, 3, 1, 1))
        
    def forward(self, x):
        # Apply 2D FFT
        fft_x = torch.fft.fft2(x, dim=(-2, -1))
        
        # Get magnitude and phase
        magnitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        # Apply frequency domain filtering
        # Emphasize certain frequency ranges for crack detection
        h, w = magnitude.shape[-2:]
        
        # Create frequency masks for different ranges
        center_h, center_w = h // 2, w // 2
        
        # High-frequency mask (edges and fine details)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=x.device), 
            torch.arange(w, device=x.device), 
            indexing='ij'
        )
        
        dist_from_center = torch.sqrt(
            (y_coords - center_h)**2 + (x_coords - center_w)**2
        )
        
        # Create frequency-selective filters
        low_freq_mask = (dist_from_center < min(h, w) * 0.1).float()
        mid_freq_mask = ((dist_from_center >= min(h, w) * 0.1) & 
                        (dist_from_center < min(h, w) * 0.3)).float()
        high_freq_mask = (dist_from_center >= min(h, w) * 0.3).float()
        
        # Apply different emphasis to different frequency bands
        filtered_magnitude = (magnitude * 
                            (0.5 * low_freq_mask + 
                             1.0 * mid_freq_mask + 
                             1.5 * high_freq_mask))
        
        # Reconstruct signal
        filtered_fft = filtered_magnitude * torch.exp(1j * phase)
        
        # Convert back to spatial domain
        filtered_spatial = torch.fft.ifft2(filtered_fft, dim=(-2, -1))
        
        # Take real part and ensure proper range
        result = torch.real(filtered_spatial)
        
        # Apply learnable weights
        result = result * self.freq_weights
        
        return result

class FusionBridge(nn.Module):
    """Physics-guided fusion bridge between spatial and frequency streams"""
    def __init__(self, channels):
        super().__init__()
        self.topology_net = nn.Sequential(
            Conv3X3(channels*2, channels),
            nn.ReLU(),
            Conv3X3(channels, 1),
            nn.Sigmoid()
        )
        self.feature_fuser = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, spatial_feat, freq_feat):
        concat = torch.cat([spatial_feat, freq_feat], dim=1)
        connectivity = self.topology_net(concat)
        connectivity = connectivity.expand(-1, spatial_feat.size(1), -1, -1)
        fused = connectivity * spatial_feat + (1 - connectivity) * freq_feat
        return self.feature_fuser(torch.cat([fused, freq_feat], dim=1))

class HNet(nn.Module):
    """Combines 2 UNet architectures with precise feature reconstruction for crack detection"""
    def __init__(self, num_classes=1):
        super().__init__()
        
        # FFT-based frequency domain preprocessing
        self.freq_preprocess = FFTProcessor()
        
        # Spatial stream
        self.spatial_enc1 = Down(3, 64)
        self.spatial_enc2 = Down(64, 128)
        self.spatial_enc3 = Down(128, 256)
        self.spatial_enc4 = Down(256, 512)
        self.spatial_bottleneck = nn.Sequential(
            Conv3X3(512, 1024),
            nn.ReLU()
        )
        
        # Frequency stream
        self.freq_enc1 = Down(3, 64)
        self.freq_enc2 = Down(64, 128)
        self.freq_enc3 = Down(128, 256)
        self.freq_enc4 = Down(256, 512)
        self.freq_bottleneck = nn.Sequential(
            Conv3X3(512, 1024),
            nn.ReLU()
        )
        
        # Bottleneck fusion
        self.fusion_bridge = FusionBridge(1024) 

        # Decoders
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
        
        # Output processing
        self.final_conv = nn.Sequential(
            Conv3X3(2, 32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )
            

    def forward(self, x):
        # Spatial encoder - uses raw input and tracks indices
        s1, s1_skip, s1_indices = self.spatial_enc1(x)
        s2, s2_skip, s2_indices = self.spatial_enc2(s1)
        s3, s3_skip, s3_indices = self.spatial_enc3(s2)
        s4, s4_skip, s4_indices = self.spatial_enc4(s3)
        s_bottleneck = self.spatial_bottleneck(s4)
        
        # Frequency encoder - uses real frequency domain input and tracks indices
        freq_input = self.freq_preprocess(x)
        f1, f1_skip, f1_indices = self.freq_enc1(freq_input)
        f2, f2_skip, f2_indices = self.freq_enc2(f1)
        f3, f3_skip, f3_indices = self.freq_enc3(f2)
        f4, f4_skip, f4_indices = self.freq_enc4(f3)
        f_bottleneck = self.freq_bottleneck(f4)
        
        # Feature fusion
        fused = self.fusion_bridge(s_bottleneck, f_bottleneck)
        
        # Spatial decoder with max unpooling
        sd1 = self.spatial_dec1(fused, s4_skip, s4_indices)
        sd2 = self.spatial_dec2(sd1, s3_skip, s3_indices)
        sd3 = self.spatial_dec3(sd2, s2_skip, s2_indices)
        sd4 = self.spatial_dec4(sd3, s1_skip, s1_indices)
        
        # Frequency decoder with max unpooling
        fd1 = self.freq_dec1(fused, f4_skip, f4_indices)
        fd2 = self.freq_dec2(fd1, f3_skip, f3_indices)
        fd3 = self.freq_dec3(fd2, f2_skip, f2_indices)
        fd4 = self.freq_dec4(fd3, f1_skip, f1_indices)
        
        # Generate decoder outputs
        spatial_out = self.spatial_out(sd4)
        freq_out = self.freq_out(fd4)
        
        # Generate final output using existing code
        spatial_prob = torch.sigmoid(spatial_out)
        freq_prob = torch.sigmoid(freq_out)
        combined = torch.cat([spatial_prob, freq_prob], dim=1)
        final_output = self.final_conv(combined)
        
        return final_output