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
        padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)  # 使用反射填充保留特征图边界信息
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


