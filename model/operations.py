import torch
import torch.nn as nn


OPS = {
    'conv_1x1': lambda C_in, C_out: ConvBlock(C_in, C_out, 1),
    'conv_3x3': lambda C_in, C_out: ConvBlock(C_in, C_out, 3),
    'conv_5x5': lambda C_in, C_out: ConvBlock(C_in, C_out, 5),
    'conv_7x7': lambda C_in, C_out: ConvBlock(C_in, C_out, 7),
    'conv_1x3': lambda C_in, C_out: ConvBlock_S(C_in, C_out, 3, mode = 1),
    'conv_3x1': lambda C_in, C_out: ConvBlock_S(C_in, C_out, 3, mode = 2),
    'conv_1x5': lambda C_in, C_out: ConvBlock_S(C_in, C_out, 5, mode = 1),
    'conv_5x1': lambda C_in, C_out: ConvBlock_S(C_in, C_out, 5, mode = 2),
    'dil_3x3': lambda C_in, C_out: DilConv(C_in, C_out, 3, stride = 1, padding = 2, dilation = 2),
    'dil_5x5': lambda C_in, C_out: DilConv(C_in, C_out, 5, stride = 1, padding = 4, dilation = 2),
    'dil_7x7': lambda C_in, C_out: DilConv(C_in, C_out, 7, stride = 1, padding = 6, dilation = 2)
}

class ConvBlock(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride=1, dilation=1, groups=1, affine=True):
        super(ConvBlock, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding=padding, bias=False, dilation=dilation, groups=groups),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
            nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.op(x)

class ConvBlock_S(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, mode, stride=1, dilation=1, groups=1, affine=True):
        super(ConvBlock_S, self).__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        if mode == 1:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_out, (1, kernel_size), stride=(1, 1), padding=(0, padding), bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=False)
            )
        elif mode == 2:
            self.op = nn.Sequential(
                nn.Conv2d(C_in, C_out, (kernel_size, 1), stride=(1, 1), padding=(padding, 0), bias=False),
                nn.BatchNorm2d(C_out, affine=affine),
                nn.ReLU(inplace=False)
            )

    def forward(self, x):
        return self.op(x)