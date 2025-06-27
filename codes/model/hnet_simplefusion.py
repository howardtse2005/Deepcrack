import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config as cfg

def Conv3X3(in_, out):
    return nn.Conv2d(in_, out, cfg.kernel_size, padding=cfg.kernel_size//2)

class UNetDownBlock(nn.Module):
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

class UNetUpBlock(nn.Module):
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
        
        # Frequency preprocessing by texture enhancement
        freq_kernel_size = int(cfg.kernel_size * 1.5) if cfg.kernel_size > 3 else cfg.kernel_size + 2
        freq_padding = freq_kernel_size // 2
        
        self.freq_preprocess = nn.Sequential(
            nn.Conv2d(3, 3, freq_kernel_size, padding=freq_padding),
            nn.ReLU()
        )
        
        # Spatial stream
        self.spatial_enc1 = UNetDownBlock(3, 64)
        self.spatial_enc2 = UNetDownBlock(64, 128)
        self.spatial_enc3 = UNetDownBlock(128, 256)
        self.spatial_enc4 = UNetDownBlock(256, 512)
        self.spatial_bottleneck = nn.Sequential(
            Conv3X3(512, 1024),
            nn.ReLU()
        )
        
        # Frequency stream
        self.freq_enc1 = UNetDownBlock(3, 64)
        self.freq_enc2 = UNetDownBlock(64, 128)
        self.freq_enc3 = UNetDownBlock(128, 256)
        self.freq_enc4 = UNetDownBlock(256, 512)
        self.freq_bottleneck = nn.Sequential(
            Conv3X3(512, 1024),
            nn.ReLU()
        )
        
        # Bottleneck fusion
        self.fusion_bridge = FusionBridge(1024) 

        # Decoders
        self.spatial_dec1 = UNetUpBlock(1024, 512)
        self.spatial_dec2 = UNetUpBlock(512, 256)
        self.spatial_dec3 = UNetUpBlock(256, 128)
        self.spatial_dec4 = UNetUpBlock(128, 64)
        self.spatial_out = nn.Conv2d(64, num_classes, 1)
        
        self.freq_dec1 = UNetUpBlock(1024, 512)
        self.freq_dec2 = UNetUpBlock(512, 256)
        self.freq_dec3 = UNetUpBlock(256, 128)
        self.freq_dec4 = UNetUpBlock(128, 64)
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