import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config as cfg

def ConvNxN(in_, out, kernel_size=3, dilation=1):
    """Flexible convolution with configurable kernel size and dilation"""
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1)) // 2
    return nn.Conv2d(in_, out, kernel_size, padding=padding, dilation=dilation)

class Down(nn.Module):
    """U-Net encoder block with standard max pooling"""
    def __init__(self, in_ch, out_ch, kernel_size=3, downsample=True, dilation=1):
        super().__init__()
        self.downsample = downsample
        if downsample:
            self.pool = nn.MaxPool2d(2, stride=2)  # No indices needed
        
        self.conv = nn.Sequential(
            ConvNxN(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            ConvNxN(out_ch, out_ch, kernel_size=3, dilation=dilation),  # Apply dilation to both convs
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        skip = x  # Save for skip connection before pooling
        if self.downsample:
            x = self.pool(x)  # Simple pooling, no indices
        x = self.conv(x)
        if self.downsample:
            return x, skip  # Only return features and skip
        return x

class CoarseDown(nn.Module):
    """Coarse stream encoder with stride=2 downsampling and dilated convolutions"""
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, downsample_factor=2):
        super().__init__()
        
        # Use stride=2 convolution for learnable downsampling (better information preservation)
        self.downsample_conv = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=downsample_factor, padding=1)
        
        self.conv = nn.Sequential(
            ConvNxN(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            ConvNxN(out_ch, out_ch, kernel_size=3, dilation=dilation),  # Apply dilation to both convs
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

    def forward(self, x):
        skip = x  # Save for skip connection before downsampling
        
        # Apply learnable 2x downsampling with stride=2
        x = self.downsample_conv(x)
        
        # Apply dilated convolutions
        x = self.conv(x)
        
        return x, skip

class Up(nn.Module):
    """Original U-Net decoder block with bilinear upsampling"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super().__init__()
        
        # Use bilinear upsampling (standard U-Net approach)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Skip connections have half the channels of the decoder input
            # After concatenation: in_ch (upsampled) + in_ch//2 (skip) = in_ch + in_ch//2
            self.conv = nn.Sequential(
                ConvNxN(in_ch + in_ch//2, out_ch, kernel_size=3),  # Handle concatenated channels
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                ConvNxN(out_ch, out_ch, kernel_size=3),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        else:
            # Alternative: transpose convolution
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
            self.conv = nn.Sequential(
                ConvNxN(in_ch, out_ch, kernel_size=3),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                ConvNxN(out_ch, out_ch, kernel_size=3),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

    def forward(self, x, skip):
        # Upsample the input
        x = self.up(x)
        
        # Handle dimension mismatches (standard U-Net approach)
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection with upsampled features
        x = torch.cat([skip, x], dim=1)
        
        # Apply double convolution
        return self.conv(x)

class Fusion(nn.Module):
    """Learnable weighted concatenation fusion for symmetric dual-stream crack detection
    Based on U-Net concatenation philosophy with learnable weight control.
    """
    def __init__(self):
        super().__init__()
        
        # Learnable weights for fine/coarse balance
        self.fine_weight = nn.Parameter(torch.tensor(0.5))
        self.coarse_weight = nn.Parameter(torch.tensor(0.5))
        
        # U-Net style fusion processing (like decoder blocks)
        self.fusion_conv = nn.Sequential(
            ConvNxN(2048, 1024, kernel_size=3),  # 2048 → 1024 channels
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            ConvNxN(1024, 1024, kernel_size=3),  # Standard double conv
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )
        
    def forward(self, fine_features, coarse_features):
        weighted_fine = self.fine_weight * fine_features
        weighted_coarse = self.coarse_weight * coarse_features

        concatenated = torch.cat([weighted_fine, weighted_coarse], dim=1)  # [B, 2048, H, W]
 
        output = self.fusion_conv(concatenated)
        
        return output

class HNet(nn.Module):
    """Multi-Scale Spatial HNet: Fine-scale + Coarse-scale streams for context-aware crack detection"""
    def __init__(self, num_classes=1):
        super().__init__()
        
        # Initial convolution blocks (like original U-Net)
        self.fine_inc = nn.Sequential(
            ConvNxN(3, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ConvNxN(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Coarse stream: 2x downsampling with large kernel for initial context
        self.coarse_inc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Learnable 2x downsample input
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ConvNxN(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Fine-scale encoder (standard 3x3 kernels for local details)
        self.fine_enc1 = Down(64, 128, kernel_size=3)   # Local patterns  
        self.fine_enc2 = Down(128, 256, kernel_size=3)  # Crack textures
        self.fine_enc3 = Down(256, 512, kernel_size=3)  # Fine features
        self.fine_enc4 = Down(512, 1024, kernel_size=3) # Deep features
        
        # Coarse-scale encoder with gradual dilation increase and stride=2 downsampling
        self.coarse_enc1 = CoarseDown(64, 128, kernel_size=5, dilation=1)   # Standard dilation at shallow level
        self.coarse_enc2 = CoarseDown(128, 256, kernel_size=5, dilation=2)  # Start expanding receptive field
        self.coarse_enc3 = CoarseDown(256, 512, kernel_size=5, dilation=4)  # Larger context capture
        self.coarse_enc4 = CoarseDown(512, 1024, kernel_size=5, dilation=8) # Maximum context understanding
        
        # Bottleneck processing (symmetric design)
        self.fine_bottleneck = nn.Sequential(
            ConvNxN(1024, 1024, kernel_size=3),
            nn.ReLU()
        )
        # Coarse stream: same depth as fine stream with final dilation
        self.coarse_bottleneck = nn.Sequential(
            ConvNxN(1024, 1024, kernel_size=3, dilation=4),  # Keep large receptive field at bottleneck
            nn.ReLU()
        )
        
        # Multi-scale fusion at bottleneck (adaptive based on gap detection)
        self.fusion_bottleneck = Fusion()
        
        # Fine-scale decoder (preserves crack details)
        self.fine_dec1 = Up(1024, 512)
        self.fine_dec2 = Up(512, 256)
        self.fine_dec3 = Up(256, 128)
        self.fine_dec4 = Up(128, 64)
        self.fine_out = nn.Conv2d(64, num_classes, 1)
        
        # Coarse-scale decoder (compensate for 2x downsampling with extra upsampling)
        self.coarse_dec1 = Up(1024, 512)
        self.coarse_dec2 = Up(512, 256)
        self.coarse_dec3 = Up(256, 128)
        self.coarse_dec4 = Up(128, 64)
        
        # Additional upsampling to match fine stream resolution (compensate for initial 2x downsample)
        self.coarse_final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvNxN(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.coarse_out = nn.Conv2d(64, num_classes, 1)
        
        # Final combination (fine details + coarse validation)
        self.final_conv = nn.Sequential(
            ConvNxN(2, 64, kernel_size=3),
            nn.ReLU(),
            ConvNxN(64, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        # Initial convolution processing
        fine_inc = self.fine_inc(x)    # [B, 64, 448, 448] - Fine initial features at full resolution
        coarse_inc = self.coarse_inc(x) # [B, 64, 224, 224] - Coarse initial features at 2x downsampled resolution
        
        # Fine-scale encoder path (local crack detection)
        f1, f1_skip = self.fine_enc1(fine_inc)  # [B, 128, 224, 224] - Local patterns
        f2, f2_skip = self.fine_enc2(f1)        # [B, 256, 112, 112] - Crack textures
        f3, f3_skip = self.fine_enc3(f2)        # [B, 512, 56, 56] - Fine features
        f4, f4_skip = self.fine_enc4(f3)        # [B, 1024, 28, 28] - Deep features
        f_bottleneck = self.fine_bottleneck(f4)              # [B, 1024, 28, 28]
        
        # Coarse-scale encoder path with stride=2 downsampling and increasing dilation
        c1, c1_skip = self.coarse_enc1(coarse_inc) # [B, 128, 112, 112] - Stride=2 downsampled, dilation=1
        c2, c2_skip = self.coarse_enc2(c1)         # [B, 256, 56, 56] - Stride=2 downsampled, dilation=2
        c3, c3_skip = self.coarse_enc3(c2)         # [B, 512, 28, 28] - Stride=2 downsampled, dilation=4
        c4, c4_skip = self.coarse_enc4(c3)         # [B, 1024, 14, 14] - Stride=2 downsampled, dilation=8
        c_bottleneck = self.coarse_bottleneck(c4)              # [B, 1024, 14, 14]
        
        # Multi-scale fusion at bottleneck (upsample coarse to match fine resolution)
        c_bottleneck_upsampled = F.interpolate(c_bottleneck, size=f_bottleneck.shape[2:], mode='bilinear', align_corners=True)
        fused_bottleneck = self.fusion_bottleneck(f_bottleneck, c_bottleneck_upsampled)
        
        # Fine-scale decoder (detail preservation)
        fd1 = self.fine_dec1(fused_bottleneck, f4_skip)  # [B, 512, 56, 56]
        fd2 = self.fine_dec2(fd1, f3_skip)               # [B, 256, 112, 112]
        fd3 = self.fine_dec3(fd2, f2_skip)               # [B, 128, 224, 224]
        fd4 = self.fine_dec4(fd3, f1_skip)               # [B, 64, 448, 448]
        
        # Coarse-scale decoder (needs to match skip connection resolutions)
        # Upsample fused bottleneck to match coarse stream's expected input size
        fused_for_coarse = F.interpolate(fused_bottleneck, size=c4_skip.shape[2:], mode='bilinear', align_corners=True)
        
        # Apply coarse decoder with appropriately sized skip connections
        cd1 = self.coarse_dec1(fused_for_coarse, c4_skip)  # [B, 512, 28, 28]
        
        # Upsample cd1 to match c3_skip size
        cd1_upsampled = F.interpolate(cd1, size=c3_skip.shape[2:], mode='bilinear', align_corners=True)
        cd2 = self.coarse_dec2(cd1_upsampled, c3_skip)     # [B, 256, 56, 56]
        
        # Continue with proper size matching
        cd2_upsampled = F.interpolate(cd2, size=c2_skip.shape[2:], mode='bilinear', align_corners=True)
        cd3 = self.coarse_dec3(cd2_upsampled, c2_skip)     # [B, 128, 112, 112]
        
        cd3_upsampled = F.interpolate(cd3, size=c1_skip.shape[2:], mode='bilinear', align_corners=True)
        cd4 = self.coarse_dec4(cd3_upsampled, c1_skip)     # [B, 64, 224, 224]
        
        # Final upsampling to match fine stream resolution
        cd4_final = self.coarse_final_upsample(cd4)        # [B, 64, 448, 448]
        
        # Generate outputs
        fine_out = self.fine_out(fd4)       # [B, 1, 448, 448] - High sensitivity crack detection
        coarse_out = self.coarse_out(cd4_final)  # [B, 1, 448, 448] - Context-aware validation
        
        # Smart combination: Fine detection + Coarse validation
        fine_prob = torch.sigmoid(fine_out)     # Crack sensitivity
        coarse_prob = torch.sigmoid(coarse_out) # Structural understanding
        
        # Combine: Keep fine detections that coarse stream validates
        combined = torch.cat([fine_prob, coarse_prob], dim=1)
        final_output = self.final_conv(combined)
        
        return final_output