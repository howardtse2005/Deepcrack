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

class RecurrentBlock(nn.Module):
    """Recurrent Convolutional Block from R2U-Net paper"""
    def __init__(self, ch_out, t=2):
        super(RecurrentBlock, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            else:
                x1 = self.conv(x + x1)
        return x1

class RRCNN_block(nn.Module):
    """Recurrent Residual Convolutional Neural Network Block"""
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            RecurrentBlock(ch_out, t=t),
            RecurrentBlock(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1

class DoubleConv(nn.Module):
    """Double convolution block replaced with RRCNN block"""
    def __init__(self, in_channels, out_channels, t=2):
        super().__init__()
        self.rrcnn = RRCNN_block(in_channels, out_channels, t=t)

    def forward(self, x):
        return self.rrcnn(x)

class Down(nn.Module):
    """Downscaling with maxpool then RRCNN block"""
    def __init__(self, in_channels, out_channels, t=2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, t=t)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then RRCNN block with attention gate"""
    def __init__(self, in_channels, out_channels, bilinear=True, t=2):
        super().__init__()

        # If bilinear, use normal upsampling
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, t=t)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, t=t)
        
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

class AttentionR2UNet(nn.Module):
    """Attention R2 U-Net: Combination of Attention U-Net and R2U-Net"""
    def __init__(self, num_classes=1, n_channels=3, bilinear=True, t=2):
        super(AttentionR2UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear
        self.t = t  # Recurrent time steps

        # Channel sizes
        factor = 2 if bilinear else 1
        base_channels = 64
        
        # Initial convolution with RRCNN block
        self.inc = DoubleConv(n_channels, base_channels, t=t)
        
        # Encoder with RRCNN blocks
        self.down1 = Down(base_channels, base_channels*2, t=t)
        self.down2 = Down(base_channels*2, base_channels*4, t=t)
        self.down3 = Down(base_channels*4, base_channels*8, t=t)
        self.down4 = Down(base_channels*8, base_channels*16 // factor, t=t)
        
        # Decoder with attention gates and RRCNN blocks
        self.up1 = Up(base_channels*16, base_channels*8 // factor, bilinear, t=t)
        self.up2 = Up(base_channels*8, base_channels*4 // factor, bilinear, t=t)
        self.up3 = Up(base_channels*4, base_channels*2 // factor, bilinear, t=t)
        self.up4 = Up(base_channels*2, base_channels, bilinear, t=t)
        
        # Output convolution
        self.outc = OutConv(base_channels, num_classes)

    def forward(self, x):
        # Encoder with recurrent residual blocks
        x1 = self.inc(x)      # [B, 64, 512, 512] - RRCNN
        x2 = self.down1(x1)   # [B, 128, 256, 256] - RRCNN
        x3 = self.down2(x2)   # [B, 256, 128, 128] - RRCNN
        x4 = self.down3(x3)   # [B, 512, 64, 64] - RRCNN
        x5 = self.down4(x4)   # [B, 1024, 32, 32] - RRCNN
        
        # Decoder with attention gates and recurrent residual blocks
        x = self.up1(x5, x4)  # [B, 512, 64, 64] - Attention + RRCNN
        x = self.up2(x, x3)   # [B, 256, 128, 128] - Attention + RRCNN
        x = self.up3(x, x2)   # [B, 128, 256, 256] - Attention + RRCNN
        x = self.up4(x, x1)   # [B, 64, 512, 512] - Attention + RRCNN
        
        # Final output
        logits = self.outc(x)
        
        return logits
