import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config as cfg

def Conv3X3(in_channels, out_channels):
    return nn.Conv2d(in_channels, out_channels, kernel_size=cfg.kernel_size, 
                     padding=cfg.kernel_size//2)

class AttentionGate(nn.Module):
    """Attention Gate module from Attention U-Net paper"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        
        return x * psi

class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            Conv3X3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            Conv3X3(out_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv with attention gate"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # If bilinear, use normal upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        
        # Attention gate
        self.attention = AttentionGate(F_g=in_channels//2, F_l=in_channels//2, F_int=in_channels//4)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Apply attention gate
        x2 = self.attention(g=x1, x=x2)
        
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
    """Original Attention U-Net from 'Attention U-Net: Learning Where to Look for the Pancreas'"""
    def __init__(self, num_classes=1, n_channels=3, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        # Channel sizes
        factor = 2 if bilinear else 1
        base_channels = 64
        
        # Initial convolution
        self.inc = DoubleConv(n_channels, base_channels)
        
        # Encoder
        self.down1 = Down(base_channels, base_channels*2)
        self.down2 = Down(base_channels*2, base_channels*4)
        self.down3 = Down(base_channels*4, base_channels*8)
        self.down4 = Down(base_channels*8, base_channels*16 // factor)
        
        # Decoder with attention gates
        self.up1 = Up(base_channels*16, base_channels*8 // factor, bilinear)
        self.up2 = Up(base_channels*8, base_channels*4 // factor, bilinear)
        self.up3 = Up(base_channels*4, base_channels*2 // factor, bilinear)
        self.up4 = Up(base_channels*2, base_channels, bilinear)
        
        # Output convolution
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # [B, 64, 512, 512]
        x2 = self.down1(x1)   # [B, 128, 256, 256]
        x3 = self.down2(x2)   # [B, 256, 128, 128]
        x4 = self.down3(x3)   # [B, 512, 64, 64]
        x5 = self.down4(x4)   # [B, 1024, 32, 32]
        
        # Decoder with attention gates
        x = self.up1(x5, x4)  # [B, 512, 64, 64] - attention gate applied to x4
        x = self.up2(x, x3)   # [B, 256, 128, 128] - attention gate applied to x3
        x = self.up3(x, x2)   # [B, 128, 256, 256] - attention gate applied to x2
        x = self.up4(x, x1)   # [B, 64, 512, 512] - attention gate applied to x1
        
        # Final output
        logits = self.outc(x)
        
        return logits
