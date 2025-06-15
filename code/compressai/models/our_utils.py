import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from compressai.layers import *
from .unet import UNet1, UNet2, UNet3


class FrequencyDecomposition(nn.Module):
    def __init__(self, in_channel=3, out_channel=3):
        super(FrequencyDecomposition, self).__init__()
        self.avgpool_3x3 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.avgpool_5x5 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.avgpool_7x7 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        self.conv1x1_l = nn.Conv2d(in_channel*3, in_channel, 1)
        self.conv1x1_h = nn.Conv2d(in_channel*3, in_channel, 1)
        self.resblock_l = ResidualBlock(in_channel, out_channel)
        self.resblock_h = ResidualBlock(in_channel, out_channel)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        identity = x
        x1 = self.avgpool_3x3(x)
        x2 = self.avgpool_5x5(x)
        x3 = self.avgpool_7x7(x)
        xa = torch.cat((x1, x2, x3), 1)
        xa = self.resblock_l(self.lrelu(self.conv1x1_l(xa)))
        x_l = xa + identity

        x4 = x - x1
        x5 = x - x2
        x6 = x - x3
        xb = torch.cat((x4, x5, x6), 1)
        xb = self.resblock_h(self.lrelu(self.conv1x1_h(xb)))
        x_h = xb + identity
        return x_l, x_h

class FrequencySynthes(nn.Module):
    def __init__(self):
        super(FrequencySynthes, self).__init__()
        self.masknet = Mask()

    def forward(self, x_l, x_h):
        mask = self.masknet(torch.cat([x_l, x_h], dim=1)).repeat([1, 3, 1, 1])
        x_hat = mask*x_l + (1.0 - mask)*x_h
        return x_hat


class Mask(nn.Module):
    def __init__(self, ch=32):
        super(Mask, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = conv(6, ch, 5, 1)
        self.conv2 = conv(ch, ch*2, 5, 1)
        self.conv3 = conv(ch*2, ch*4, 3, 1)
        self.bottleneck = conv(ch*4, ch*4, 3, 1)
        self.deconv1 = conv(ch*8, ch*4 ,3, 1)
        self.deconv2 = conv(ch*4+ch*2, ch*2, 5, 1)
        self.deconv3 = conv(ch*2+ch, ch, 5, 1)
        self.conv4 = conv(ch, 1, 1, 1)
        # self.conv4 = conv(ch, 1, 5, 1)
    
    def forward(self, x):
        x = self.conv1(x)
        conv1 = F.relu(x)
        x = self.pool(conv1)

        x = self.conv2(x)
        conv2 = F.relu(x)
        x = self.pool(conv2)

        x = self.conv3(x)
        conv3 = F.relu(x)
        x = self.pool(conv3)

        x = self.bottleneck(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv3], dim=1)
        x = self.deconv1(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv2], dim=1)
        x = self.deconv2(x)
        x = F.relu(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.cat([x, conv1], dim=1)
        x = self.deconv3(x)
        x = F.relu(x)

        mask = torch.sigmoid(self.conv4(x))
        return mask


class AINNComp(nn.Module):
    def __init__(self, M):
        super(AINNComp, self).__init__()
        self.in_nc = 3
        self.out_nc = M

        self.operations1 = nn.ModuleList()
        a = SqueezeLayer(2)
        self.operations1.append(a)
        self.in_nc *= 4
        self.inv_conv_1 = InvertibleConv1x1(self.in_nc)
        self.operations1.append(self.inv_conv_1)
        a = CouplingLayer(self.in_nc, self.in_nc, 5)
        self.operations1.append(a)
        a = CouplingLayer(self.in_nc, self.in_nc, 5)
        self.operations1.append(a)
        a = CouplingLayer(self.in_nc, self.in_nc, 5)
        self.operations1.append(a)

        # denoising1
        self.denoising1_1 = UNet3(self.in_nc, self.in_nc)
        self.denoising1_2 = UNet3(self.in_nc, self.in_nc)

        # 2nd level
        self.operations2 = nn.ModuleList()
        b = SqueezeLayer(2)
        self.operations2.append(b)
        self.in_nc *= 4
        self.inv_conv_2 = InvertibleConv1x1(self.in_nc)
        self.operations2.append(self.inv_conv_2)
        b = CouplingLayer(self.in_nc, self.in_nc, 5)
        self.operations2.append(b)
        b = CouplingLayer(self.in_nc, self.in_nc, 5)
        self.operations2.append(b)
        b = CouplingLayer(self.in_nc, self.in_nc, 5)
        self.operations2.append(b)

        # denoising2
        self.denoising2_1 = UNet2(self.in_nc, self.in_nc)
        self.denoising2_2 = UNet2(self.in_nc, self.in_nc)

        # 3rd level
        self.operations3 = nn.ModuleList()
        c = SqueezeLayer(2)
        self.operations3.append(c)
        self.in_nc *= 4
        self.inv_conv_3 = InvertibleConv1x1(self.in_nc)
        self.operations3.append(self.inv_conv_3)
        c = CouplingLayer(self.in_nc, self.in_nc, 3)
        self.operations3.append(c)
        c = CouplingLayer(self.in_nc, self.in_nc, 3)
        self.operations3.append(c)
        c = CouplingLayer(self.in_nc, self.in_nc, 3)
        self.operations3.append(c)

        # denoising3
        self.denoising3_1 = UNet1(self.in_nc, self.in_nc)
        self.denoising3_2 = UNet1(self.in_nc, self.in_nc)

        # 4th level
        self.operations4 = nn.ModuleList()
        d = SqueezeLayer(2)
        self.operations4.append(d)
        self.in_nc *= 4
        self.inv_conv_4 = InvertibleConv1x1(self.in_nc)
        self.operations4.append(self.inv_conv_4)
        d = CouplingLayer(self.in_nc, self.in_nc, 3)
        self.operations4.append(d)
        d = CouplingLayer(self.in_nc, self.in_nc, 3)
        self.operations4.append(d)
        d = CouplingLayer(self.in_nc, self.in_nc, 3)
        self.operations4.append(d)

        self.freq_D = FrequencyDecomposition()
        self.freq_S = FrequencySynthes()

        self.rdb_last_1 = RDB(self.out_nc, self.in_nc)
        self.rdb_last_2 = RDB(self.out_nc, self.in_nc)

        self.d0=0.2
        self.d1=0.2
        self.d2=0.2
        self.d3=0.2

    def forward(self, x, rev=False):
        if not rev:
            x1, x2 = self.freq_D(x)

            for op in self.operations1:
                x1, x2 = op.forward(x1, x2, False)
            for op in self.operations2:
                x1, x2 = op.forward(x1, x2, False)
            for op in self.operations3:
                x1, x2 = op.forward(x1, x2, False)
            for op in self.operations4:
                x1, x2 = op.forward(x1, x2, False)

            x = torch.cat([x1, x2], dim=1)
            b, c, h, w = x.size()
            x = torch.mean(x.view(b, c//self.out_nc, self.out_nc, h, w), dim=1)
        else:
            x1 = self.rdb_last_1(x)
            x2 = self.rdb_last_2(x)

            for op in reversed(self.operations4):
                x1, x2 = op.forward(x1, x2, True)
            x1 = x1 + self.d3 * self.denoising3_1(x1)
            x2 = x2 + self.d3 * self.denoising3_2(x2)

            for op in reversed(self.operations3):
                x1, x2 = op.forward(x1, x2, True)
            x1 = x1 + self.d2 * self.denoising2_1(x1)
            x2 = x2 + self.d2 * self.denoising2_2(x2)

            for op in reversed(self.operations2):
                x1, x2 = op.forward(x1, x2, True)
            x1 = x1 + self.d1 * self.denoising1_1(x1)
            x2 = x2 + self.d1 * self.denoising1_2(x2)

            for op in reversed(self.operations1):
                x1, x2 = op.forward(x1, x2, True)            
            x = self.freq_S(x1, x2)
        return x


class CouplingLayer(nn.Module):
    def __init__(self, split_len1, split_len2, kernal_size, clamp=1.0):
        super(CouplingLayer, self).__init__()
        self.split_len1 = split_len1
        self.split_len2 = split_len2
        self.clamp = clamp

        self.G1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.G2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)
        self.H1 = Bottleneck(self.split_len1, self.split_len2, kernal_size)
        self.H2 = Bottleneck(self.split_len2, self.split_len1, kernal_size)

    def forward(self, x1, x2, rev=False):
        if not rev:
            y1 = x1.mul(torch.exp( self.clamp * (torch.sigmoid(self.G2(x2)) * 2 - 1) )) + self.H2(x2)
            y2 = x2.mul(torch.exp( self.clamp * (torch.sigmoid(self.G1(y1)) * 2 - 1) )) + self.H1(y1)
        else:
            y2 = (x2 - self.H1(x1)).div(torch.exp( self.clamp * (torch.sigmoid(self.G1(x1)) * 2 - 1) ))
            y1 = (x1 - self.H2(y2)).div(torch.exp( self.clamp * (torch.sigmoid(self.G2(y2)) * 2 - 1) ))
        return y1, y2

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Bottleneck, self).__init__()
        # P = ((S-1)*W-S+F)/2, with F = filter size, S = stride
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights_xavier([self.conv1, self.conv2], 0.1)
        initialize_weights(self.conv3, 0)
        
    def forward(self, x):
        conv1 = self.lrelu(self.conv1(x))
        conv2 = self.lrelu(self.conv2(conv1))
        conv3 = self.conv3(conv2)
        return conv3


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

class ResBlock(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size=3, stride=1):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(inputchannel, outputchannel,
                            kernel_size, stride, padding=kernel_size//2)
        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel,
                            kernel_size, stride, padding=kernel_size//2)
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        self.conv2.weight.data=self.conv2.weight.data*0.0
        torch.nn.init.constant_(self.conv2.bias.data, 0.0)

    def forward(self, x):
        x_1 = self.relu1(x)
        firstlayer = self.conv1(x_1)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        return x + seclayer


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size//2), bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels=192, out_channels=192*4, kernel_size=3):
        super(RDB, self).__init__()
        num_layers=out_channels // in_channels - 1
        growth_rate = in_channels
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate, kernel_size=kernel_size) for i in range(num_layers)])
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, out_channels, kernel_size=1)  # kernel_size=1
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.skip(x) + self.lff(self.layers(x))

class SqueezeLayer(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, input1, input2, reverse=False):
        if not reverse:
            output1 = self.squeeze2d(input1, self.factor)  # Squeeze in forward
            output2 = self.squeeze2d(input2, self.factor)
            return output1, output2
        else:
            output1 = self.unsqueeze2d(input1, self.factor)
            output2 = self.unsqueeze2d(input2, self.factor)
            return output1, output2
        
    def jacobian(self, x, rev=False):
        return 0
        
    @staticmethod
    def squeeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert H % factor == 0 and W % factor == 0, "{}".format((H, W, factor))
        x = input.view(B, C, H // factor, factor, W // factor, factor)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
        x = x.view(B, factor * factor * C, H // factor, W // factor)
        return x

    @staticmethod
    def unsqueeze2d(input, factor=2):
        assert factor >= 1 and isinstance(factor, int)
        factor2 = factor ** 2
        if factor == 1:
            return input
        size = input.size()
        B = size[0]
        C = size[1]
        H = size[2]
        W = size[3]
        assert C % (factor2) == 0, "{}".format(C)
        x = input.view(B, factor, factor, C // factor2, H, W)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()
        x = x.view(B, C // (factor2), H * factor, W * factor)
        return x

class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        w_shape = [num_channels, num_channels]
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)
        self.register_parameter("weight", nn.Parameter(torch.Tensor(w_init)))
        self.w_shape = w_shape

    def get_weight(self, input, reverse):
        w_shape = self.w_shape
        if not reverse:
            weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)
        else:
            weight = torch.inverse(self.weight.double()).float() \
                .view(w_shape[0], w_shape[1], 1, 1)
        return weight

    def forward(self, input1, input2, reverse=False):
        weight1 = self.get_weight(input1, reverse)
        weight2 = self.get_weight(input2, reverse)
        if not reverse:
            z1 = F.conv2d(input1, weight1)
            z2 = F.conv2d(input2, weight2)
            return z1, z2
        else:
            z1 = F.conv2d(input1, weight1)
            z2 = F.conv2d(input2, weight2)
            return z1, z2


class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if init == 'xavier':
            initialize_weights_xavier([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        else:
            initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4], 0.1)
        initialize_weights(self.conv5, 0)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

def initialize_weights_xavier(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)
