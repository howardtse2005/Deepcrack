import torch
import torch.nn as nn
import torch.nn.functional as F
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

class AdaptiveFusionBridge(nn.Module):
    """Adaptive physics-guided fusion bridge that can handle different channel dimensions"""
    def __init__(self, channels, output_channels=None):
        super().__init__()
        self.channels = channels
        self.output_channels = output_channels if output_channels is not None else channels
        
        # Topology network for determining connectivity
        self.topology_net = nn.Sequential(
            Conv3X3(channels*2, channels),
            nn.ReLU(),
            Conv3X3(channels, 1),
            nn.Sigmoid()
        )
        
        # Feature fusion network
        self.feature_fuser = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        # Output projection to desired number of channels
        if self.output_channels != channels:
            self.output_proj = nn.Conv2d(channels, self.output_channels, 1)
        else:
            self.output_proj = nn.Identity()

    def forward(self, spatial_feat, freq_feat):
        concat = torch.cat([spatial_feat, freq_feat], dim=1)
        connectivity = self.topology_net(concat)
        connectivity = connectivity.expand(-1, spatial_feat.size(1), -1, -1)
        fused = connectivity * spatial_feat + (1 - connectivity) * freq_feat
        enhanced = self.feature_fuser(torch.cat([fused, freq_feat], dim=1))
        return self.output_proj(enhanced)

class HNet(nn.Module):
    """Combines 2 UNet architectures with precise feature reconstruction for crack detection"""
    def __init__(self, num_classes=1):
        super().__init__()
        
        # Frequency preprocessing by texture enhancement
        freq_kernel_size = int(cfg.kernel_size * 1.5) if cfg.kernel_size > 3 else cfg.kernel_size + 2
        freq_padding = freq_kernel_size // 2
        
        self.freq_preprocess = nn.Sequential(
            nn.Conv2d(3, 3, freq_kernel_size, padding=freq_padding),
            nn.ReLU()
        )
        
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
        
        # Multi-level fusion bridges
        self.bottleneck_fusion = AdaptiveFusionBridge(1024, output_channels=1024)  # Bottleneck fusion
        self.level4_fusion = AdaptiveFusionBridge(512, output_channels=512)         # Level 4 fusion
        self.level3_fusion = AdaptiveFusionBridge(256, output_channels=256)         # Level 3 fusion
        self.level2_fusion = AdaptiveFusionBridge(128, output_channels=128)         # Level 2 fusion
        self.level1_fusion = AdaptiveFusionBridge(64, output_channels=64)           # Level 1 fusion 

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
        
        # Frequency encoder - uses enhanced texture input and tracks indices
        freq_input = self.freq_preprocess(x)
        f1, f1_skip, f1_indices = self.freq_enc1(freq_input)
        f2, f2_skip, f2_indices = self.freq_enc2(f1)
        f3, f3_skip, f3_indices = self.freq_enc3(f2)
        f4, f4_skip, f4_indices = self.freq_enc4(f3)
        f_bottleneck = self.freq_bottleneck(f4)
        
        # Multi-level feature fusion
        fused_bottleneck = self.bottleneck_fusion(s_bottleneck, f_bottleneck)
        fused_level4 = self.level4_fusion(s4_skip, f4_skip)
        fused_level3 = self.level3_fusion(s3_skip, f3_skip)
        fused_level2 = self.level2_fusion(s2_skip, f2_skip)
        fused_level1 = self.level1_fusion(s1_skip, f1_skip)
        
        # Spatial decoder with fused features as skip connections
        sd1 = self.spatial_dec1(fused_bottleneck, fused_level4, s4_indices)
        sd2 = self.spatial_dec2(sd1, fused_level3, s3_indices)
        sd3 = self.spatial_dec3(sd2, fused_level2, s2_indices)
        sd4 = self.spatial_dec4(sd3, fused_level1, s1_indices)
        
        # Frequency decoder with fused features as skip connections
        fd1 = self.freq_dec1(fused_bottleneck, fused_level4, f4_indices)
        fd2 = self.freq_dec2(fd1, fused_level3, f3_indices)
        fd3 = self.freq_dec3(fd2, fused_level2, f2_indices)
        fd4 = self.freq_dec4(fd3, fused_level1, f1_indices)
        
        # Generate decoder outputs
        spatial_out = self.spatial_out(sd4)
        freq_out = self.freq_out(fd4)
        
        # Generate final output using existing code
        spatial_prob = torch.sigmoid(spatial_out)
        freq_prob = torch.sigmoid(freq_out)
        combined = torch.cat([spatial_prob, freq_prob], dim=1)
        final_output = self.final_conv(combined)
        
        return final_output