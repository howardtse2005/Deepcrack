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

class MultiscaleFusion(nn.Module):
    """Fusion layer connecting encoder and decoder features at multiple scales"""
    def __init__(self, scale, in_channels):
        super().__init__()
        self.scale = scale
        self.conv = nn.Sequential(
            Conv3X3(in_channels, 64),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.output = nn.Conv2d(64, 1, 1)
        
    def forward(self, encoder_feat, decoder_feat):
        fused = torch.cat([encoder_feat, decoder_feat], dim=1)
        features = self.conv(fused)
        if self.scale > 1:
            features = F.interpolate(features, scale_factor=self.scale, mode='bilinear', align_corners=False)
        return self.output(features)

class DirectionalConv(nn.Module):
    """Directional convolution to maintain linear crack structures"""
    def __init__(self, channels):
        super().__init__()
        # Create directional kernels for 4 orientations (0°, 45°, 90°, 135°)
        self.dir_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels//4)
            for _ in range(4)
        ])
        self.merge = nn.Conv2d(channels*4, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Initialize directional kernels
        self._init_directional_kernels()
        
    def _init_directional_kernels(self):
        # Initialize with directional patterns that preserve line continuity
        horizontal = torch.zeros(1, 1, 5, 5)
        horizontal[0, 0, 2, :] = 1.0  # Horizontal line
        
        vertical = torch.zeros(1, 1, 5, 5)
        vertical[0, 0, :, 2] = 1.0    # Vertical line
        
        diag1 = torch.zeros(1, 1, 5, 5)
        for i in range(5):
            diag1[0, 0, i, i] = 1.0   # 45° diagonal
            
        diag2 = torch.zeros(1, 1, 5, 5)
        for i in range(5):
            diag2[0, 0, i, 4-i] = 1.0 # 135° diagonal
            
        # Normalize kernels
        kernels = [horizontal, vertical, diag1, diag2]
        kernels = [k / k.sum() for k in kernels]
        
        # Set as initial weights (will be learnable)
        with torch.no_grad():
            for i, conv in enumerate(self.dir_convs):
                channels = conv.weight.size(0)
                for c in range(channels):
                    conv.weight[c, 0] = kernels[i][0, 0]
    
    def forward(self, x):
        # Apply directional convolutions
        dir_outputs = []
        for conv in self.dir_convs:
            dir_outputs.append(conv(x))
        
        # Merge directional features
        merged = torch.cat(dir_outputs, dim=1)
        out = self.merge(merged)
        return self.relu(out)

class ContinuityPreservingModule(nn.Module):
    """Module to enhance continuity in thin crack structures"""
    def __init__(self, in_channels):
        super().__init__()
        self.dir_conv = DirectionalConv(in_channels)
        self.continuity_gate = nn.Sequential(
            Conv3X3(in_channels, in_channels//4),
            nn.ReLU(),
            Conv3X3(in_channels//4, 1),
            nn.Sigmoid()
        )
        self.smooth_conv = Conv3X3(in_channels, in_channels)
        
    def forward(self, x):
        dir_features = self.dir_conv(x)
        continuity_mask = self.continuity_gate(x)
        
        # Apply the global continuity weight from config to scale the mask
        continuity_mask = continuity_mask * cfg.continuity_weight
        
        enhanced = x * (1-continuity_mask) + dir_features * continuity_mask
        return self.smooth_conv(enhanced)

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
        
        # Fusion
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
        
        # Noise suppression
        self.noise_head = nn.Sequential(
            Conv3X3(64, 32),
            nn.ReLU(),
            nn.Conv2d(32, num_classes, 1)
        )
        
        # Compatibility heads
        self.fuse_heads = nn.ModuleList([
            nn.Conv2d(64, 1, 1) for _ in range(5)
        ])
        
        # Spatial stream fusions
        self.spatial_fuse4 = MultiscaleFusion(scale=8, in_channels=512+512)  # s4_skip + sd1
        self.spatial_fuse3 = MultiscaleFusion(scale=4, in_channels=256+256)  # s3_skip + sd2
        self.spatial_fuse2 = MultiscaleFusion(scale=2, in_channels=128+128)  # s2_skip + sd3
        self.spatial_fuse1 = MultiscaleFusion(scale=1, in_channels=64+64)    # s1_skip + sd4
        
        # Frequency stream fusions
        self.freq_fuse4 = MultiscaleFusion(scale=8, in_channels=512+512)     # f4_skip + fd1
        self.freq_fuse3 = MultiscaleFusion(scale=4, in_channels=256+256)     # f3_skip + fd2
        self.freq_fuse2 = MultiscaleFusion(scale=2, in_channels=128+128)     # f2_skip + fd3
        self.freq_fuse1 = MultiscaleFusion(scale=1, in_channels=64+64)       # f1_skip + fd4
        
        # Cross-stream fusion (optional but powerful)
        self.cross_fusion = MultiscaleFusion(scale=1, in_channels=64+64)     # sd4 + fd4
        
        # Add continuity-preserving modules
        if hasattr(cfg, 'directional_filters') and cfg.directional_filters:
            self.spatial_continuity = ContinuityPreservingModule(64)
            self.freq_continuity = ContinuityPreservingModule(64)
        else:
            self.spatial_continuity = nn.Identity()
            self.freq_continuity = nn.Identity()
            

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
        
        # Apply continuity enhancement before final output generation
        sd4 = self.spatial_continuity(sd4)
        fd4 = self.freq_continuity(fd4)
        
        spatial_out = self.spatial_out(sd4)
        freq_out = self.freq_out(fd4)
        
        # Generate multi-scale fusion outputs for direct supervision
        spatial_fuse4_out = self.spatial_fuse4(s4_skip, sd1)
        spatial_fuse3_out = self.spatial_fuse3(s3_skip, sd2)
        # spatial_fuse2_out = self.spatial_fuse2(s2_skip, sd3)
        # spatial_fuse1_out = self.spatial_fuse1(s1_skip, sd4)
        
        
        # freq_fuse4_out = self.freq_fuse4(f4_skip, fd1)
        # freq_fuse3_out = self.freq_fuse3(f3_skip, fd2)
        freq_fuse2_out = self.freq_fuse2(f2_skip, fd3)
        freq_fuse1_out = self.freq_fuse1(f1_skip, fd4)
        
        # Cross-stream fusion
        cross_out = self.cross_fusion(sd4, fd4)
        
        # Generate final output using existing code
        spatial_prob = torch.sigmoid(spatial_out)
        freq_prob = torch.sigmoid(freq_out)
        combined = torch.cat([spatial_prob, freq_prob], dim=1)
        final_output = self.final_conv(combined)
        
        # Noise suppression
        noise_map = self.noise_head(fd4)
        final_output = final_output - cfg.noise_suppression_weight * noise_map
        
        # Compatibility outputs
        fuse1 = self.fuse_heads[0](sd4)
        fuse2 = self.fuse_heads[1](sd4)
        fuse3 = self.fuse_heads[2](fd4)
        fuse4 = self.fuse_heads[3](fd4)
        fuse5 = self.fuse_heads[4](fd4)
        
        # Replace previous fuse outputs with new multi-scale fusion outputs
        fuse5 = cross_out           # Cross-stream fusion
        fuse4 = spatial_fuse4_out   # Deepest spatial fusion
        fuse3 = spatial_fuse3_out   # Mid-level spatial fusion
        fuse2 = freq_fuse2_out      # Mid-level frequency fusion
        fuse1 = freq_fuse1_out      # Shallow frequency fusion
        
        return final_output, fuse5, fuse4, fuse3, fuse2, fuse1