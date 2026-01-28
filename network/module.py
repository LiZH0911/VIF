import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------------------------------------------------------------------------------------------------------------
class ConvBnLeakyRelu2d(nn.Module):
    # convolution
    # batch normalization
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False, negative_slope=0.2, inplace=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        )
    def forward(self, x):
        return self.net(x)

class ConvLeakyRelu2d(nn.Module):
    # convolution(bias=True)
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=True, negative_slope=0.2, inplace=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=inplace)
        )
    def forward(self, x):
        return self.net(x)

class ConvBnTanh2d(nn.Module):
    # convolution
    # batch normalization
    # tanh
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1, bias=False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.Tanh()
        )
    def forward(self,x):
        x = self.net(x)
        return x / 2 + 0.5  # [0,1]

class Conv1(nn.Module):
    # 1*1 convolution
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)

class ConvReflectPad(nn.Module):
    # reflection_padding
    # convolution(padding=0)
    # batch normalization
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False):
        super().__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)  # 使用反射填充保留特征图边界信息
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=0, stride=stride, dilation=dilation, bias=bias), # 卷积
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

# ------------------------------------------------------------------------------------------------------------
class DenseConv(nn.Module):
    def __init__(self,in_channels, out_channels):
        super().__init__()
        self.conv = ConvReflectPad(in_channels, out_channels)
    def forward(self,x):
        x = torch.cat((x,self.conv(x)),dim=1)
        return x

class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, dense_out):
        super().__init__()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.add_module('dense_conv' + str(i), # 模块名称
                            DenseConv(in_channels + i * out_channels, out_channels))
        self.adjust_conv = ConvReflectPad(in_channels + num_layers * out_channels, dense_out)
    def forward(self, x):
        for i in range(self.num_layers):
            dense_conv = getattr(self, 'dense_conv' + str(i))
            x = dense_conv(x)
        x = self.adjust_conv(x)
        return x

# ------------------------------------------------------------------------------------------------------------
# SE(Squeeze-and-Excitation Networks)
# 通道注意力：在通道维度上为每个通道分配权重，强调有用的通道特征，抑制无用通道
# 作用: 增强语义相关的通道响应（例如边缘、纹理或高层语义通道）
class SEBlock(nn.Module):
    """SE: Squeeze-and-Excitation Networks

    Reference: "Squeeze-and-Excitation Networks"
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x).view(b, c)          # squeeze 全局平均池化GAP
        y = self.fc(y).view(b, c, 1, 1)          # excite 全连接层生成通道注意力，为每个通道分配权重
        return x * y.expand_as(x)                # scale 按通道缩放

class ConvSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvSEBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.se = SEBlock(out_ch, reduction=16)

    def forward(self, x):
        x = self.conv(x)
        x = self.se(x)
        return x

# ------------------------------------------------------------------------------------------------------------
# CBAM(Convolutional Block Attention Module)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # along channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM)

    Reference: "CBAM: Convolutional Block Attention Module"
    Usage: cbam = CBAM(channels); out = cbam(feature_map)
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super().__init__()
        self.channel_att = ChannelAttention(channels, ratio=reduction)
        self.spatial_att = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class ConvCBAMBlock(nn.Module):
    def __init__(self, in_ch, out_ch, reduction=16, kernel=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.cbam = CBAM(out_ch, reduction=reduction)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        return x

# ------------------------------------------------------------------------------------------------------------
def get_eca_kernel_size(channels, gamma=2, b=1):
    t = int(abs((math.log2(channels) + gamma) / b))
    if t % 2 == 0:
        t += 1
    return max(t, 1)

class ECA(nn.Module):
    """ECA-Net: Efficient Channel Attention

    Reference: "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"
    Uses a 1D conv across channels (no FC) with adaptive kernel size.
    """

    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.channels = channels
        k = get_eca_kernel_size(channels, gamma, b)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.size()
        y = self.avg_pool(x)        # (B, C, 1, 1)
        y = y.view(b, 1, c)         # (B, 1, C) for Conv1d
        y = self.conv(y)            # (B, 1, C)
        y = self.sigmoid(y)
        y = y.view(b, c, 1, 1)
        return x * y.expand_as(x)