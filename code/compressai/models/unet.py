import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits    
    
# my 1次down up
class UNet1(nn.Module):
    def __init__(self, in_channels=192, out_channels=192, bilinear=True):
        super(UNet1, self).__init__()
        self.n_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 384)
        self.down1 = Down(384, 384)
        self.up1 = Up(384*2, 192, bilinear)
        self.outc = OutConv(192, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x = self.up1(x2, x1)
        logits = self.outc(x)
        # return logits * 0.2 + identity
        return logits + identity

# my 2次down up
class UNet2(nn.Module):
    def __init__(self, in_channels=48, out_channels=48, bilinear=True):
        super(UNet2, self).__init__()
        self.n_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 192)
        self.down1 = Down(192, 384)
        self.down2 = Down(384, 384)
        self.up1 = Up(384*2, 192, bilinear)
        self.up2 = Up(192*2, 96, bilinear)
        self.outc = OutConv(96, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        logits = self.outc(x)
        # return logits * 0.2 + identity
        return logits + identity

# my 3次down up
class UNet3(nn.Module):
    def __init__(self, in_channels=12, out_channels=12, bilinear=True):
        super(UNet3, self).__init__()
        self.n_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 96)
        self.down1 = Down(96, 192)
        self.down2 = Down(192, 384)
        self.down3 = Down(384, 384)
        self.up1 = Up(384*2, 192, bilinear)
        self.up2 = Up(192*2, 96, bilinear)
        self.up3 = Up(96*2, 48, bilinear)
        self.outc = OutConv(48, out_channels)

    def forward(self, x):
        identity = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        # return logits * 0.2 + identity
        return logits + identity

if __name__ == "__main__":
    # Demo Test

    # input_image = torch.ones([1,3,256,256]) 
    # unet = UNet(3, 10)
    # z = unet(input_image)
    # print(unet)
    # print(z.shape)        # torch.Size([1, 10, 256, 256])
    
    input_image1 = torch.ones([1,192,32,32]) 
    unet1 = UNet1()
    z = unet1(input_image1)
    print(unet1)
    print(z.shape)          # torch.Size([1, 192, 32, 32])

    # input_image2 = torch.ones([1,48,64,64]) 
    # unet2 = UNet2()
    # z = unet2(input_image2)
    # print(unet2)
    # print(z.shape)          # torch.Size([1, 48, 64, 64])

    # input_image3 = torch.ones([1,12,128,128]) 
    # unet3 = UNet3()
    # z = unet3(input_image3)
    # print(unet3)
    # print(z.shape)          # torch.Size([1, 12, 128, 128])