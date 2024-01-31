import time
from collections import namedtuple
import numpy as np
import torch.nn.functional as F
from .operations import *
from .genotypes import *

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

class MixedOp(nn.Module):
    def __init__(self, C_in, C_out, weights):
        super(MixedOp, self).__init__()
        k = 0.1
        self._ops = nn.ModuleList()
        self._weight = []
        sum1 = 0
        for primitive in weights:
            if primitive[1]>k:
                sum1 += primitive[1]
        for primitive in weights:
            if primitive[1]>k:
                op = OPS[primitive[0]](C_in, C_out)
                self._ops.append(op)
                self._weight.append(primitive[1]/sum1)

    def forward(self, x):
        a = []
        for w, op in zip(self._weight, self._ops):
            if w != 0:
                a.append(w*op(x))
            else: 
                pass
        return sum(a)

class AttentionModel(nn.Module):
    def __init__(self, weights):
        super(AttentionModel, self).__init__()

        self.MaxP = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.AvgP = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.Node_x_1 = MixedOp(1, 16, weights[0])
        self.Node_x_2 = MixedOp(16, 32, weights[2])
        self.Node_x_3 = MixedOp(96, 64, weights[4])
        self.Node_x_4 = MixedOp(64, 32, weights[6])
        self.Node_x_5 = MixedOp(32, 16, weights[8])

        self.Node_y_1 = MixedOp(1, 16, weights[1])
        self.Node_y_2 = MixedOp(16, 32, weights[3])
        self.Node_y_3 = MixedOp(96, 64, weights[5])
        self.Node_y_4 = MixedOp(64, 32, weights[7])
        self.Node_y_5 = MixedOp(32, 16, weights[9])

        self.dil_xtoy = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=False)
        )
        self.dil_ytox = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=2, dilation=2,
                      groups=32, bias=False),
            nn.BatchNorm2d(32, affine=True),
            nn.ReLU(inplace=False)
        )

    def forward(self, under, over):

        x_1 = self.Node_x_1(under)
        y_1 = self.Node_y_1(over)
        
        x_2 = self.Node_x_2(x_1)
        y_2 = self.Node_y_2(y_1)

        x_2_down = torch.cat([self.MaxP(x_2), self.AvgP(x_2)], dim=1)
        y_2_down = torch.cat([self.MaxP(y_2), self.AvgP(y_2)], dim=1)

        x_2_dil = self.dil_ytox(y_2)
        y_2_dil = self.dil_xtoy(x_2)

        x_3_down = self.Node_x_3(torch.cat([x_2_down, x_2_dil], dim=1))
        y_3_down = self.Node_y_3(torch.cat([y_2_down, y_2_dil], dim=1))

        x_4 = self.Node_x_4(F.interpolate(x_3_down, size=under.shape[-2:], mode='bilinear')) + x_2 
        y_4 = self.Node_y_4(F.interpolate(y_3_down, size=under.shape[-2:], mode='bilinear')) + y_2

        A_x = self.Node_x_5(x_4)
        A_y = self.Node_y_5(y_4)

        F_i = A_x * x_1 + A_y * y_1

        return F_i, A_x, A_y

class RetinexModel(nn.Module):
    def __init__(self, weights):
        super(RetinexModel, self).__init__()

        self.Node_div = MixedOp(16, 16, weights[0])
        self.Node_i_1 = MixedOp(16, 32, weights[1])
        self.Node_i_2 = MixedOp(32, 16, weights[2])
        self.Node_l_1 = MixedOp(16, 32, weights[3])
        self.Node_l_2 = MixedOp(32, 16, weights[4])
        self.reconstruction = conv_layer(16, 1, 3)
        self.tanh = nn.Tanh()
    
    def forward(self, F_i):
        
        i = self.Node_div(F_i)

        l = F_i / (torch.sigmoid(i) + 0.00001)

        i1 = self.Node_i_2(self.Node_i_1(i)) + i
        i2 = self.Node_i_2(self.Node_i_1(i1)) + i1
        i3 = self.Node_i_2(self.Node_i_1(i2)) + i2

        l1 = l / (torch.sigmoid(self.Node_l_2(self.Node_l_1(l))) + 0.00001)
        l2 = l1 / (torch.sigmoid(self.Node_l_2(self.Node_l_1(l1))) + 0.00001)
        l3 = l2 / (torch.sigmoid(self.Node_l_2(self.Node_l_1(l2))) + 0.00001)

        F_f = i3 * l3
        F = self.tanh(self.reconstruction(F_f))

        return F

class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        arch_genotype = eval("arch")

        self.att = AttentionModel(arch_genotype.v_att)
        self.ret = RetinexModel(arch_genotype.v_ret)

    def forward(self, input):
        o = input[:,0,:,:].unsqueeze(dim=1)
        u = input[:,1,:,:].unsqueeze(dim=1)

        f_i_feature, _x1, _x2 = self.att(u, o)
        f = self.ret(f_i_feature)

        return f
