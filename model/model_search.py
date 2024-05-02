import torch
from collections import namedtuple
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from .operations import *
from .genotypes import SPACE_SIMPLE
from loss.loss import GradientLoss, PerceptualLoss, ContrastiveLoss, TVLoss, LossFunctionlcolor
from loss.mef_ssim import MEFSSIM
from kornia.losses import PSNRLoss, SSIMLoss
import itertools

def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)

class MixedOp(nn.Module):
    def __init__(self, C_in, C_out):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in SPACE_SIMPLE:
            op = OPS[primitive](C_in, C_out)
            self._ops.append(op)

    def forward(self, x, weights):
        a = []
        for w, op in zip(weights, self._ops):
            if w != 0:
                a.append(w*op(x))
            else: 
                pass
        return sum(a)

class AttentionModel(nn.Module):
    def __init__(self):
        super(AttentionModel, self).__init__()

        self.MaxP = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.AvgP = nn.AvgPool2d(2, stride=2, ceil_mode=True)

        self.Node_x_1 = MixedOp(1, 16)
        self.Node_x_2 = MixedOp(16, 32)
        self.Node_x_3 = MixedOp(96, 64)
        self.Node_x_4 = MixedOp(64, 32)
        self.Node_x_5 = MixedOp(32, 16)

        self.Node_y_1 = MixedOp(1, 16)
        self.Node_y_2 = MixedOp(16, 32)
        self.Node_y_3 = MixedOp(96, 64)
        self.Node_y_4 = MixedOp(64, 32)
        self.Node_y_5 = MixedOp(32, 16)

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

    def forward(self, under, over, weights):

        x_1 = self.Node_x_1(under, weights[0])
        y_1 = self.Node_y_1(over, weights[1])
        
        x_2 = self.Node_x_2(x_1, weights[2])
        y_2 = self.Node_y_2(y_1, weights[3])

        x_2_down = torch.cat([self.MaxP(x_2), self.AvgP(x_2)], dim=1)
        y_2_down = torch.cat([self.MaxP(y_2), self.AvgP(y_2)], dim=1)

        x_2_dil = self.dil_ytox(y_2)
        y_2_dil = self.dil_xtoy(x_2)

        x_3_down = self.Node_x_3(torch.cat([x_2_down, x_2_dil], dim=1), weights[4])
        y_3_down = self.Node_y_3(torch.cat([y_2_down, y_2_dil], dim=1), weights[5])

        x_4 = self.Node_x_4(F.interpolate(x_3_down, size=under.shape[-2:], mode='bilinear'), weights[6]) + x_2
        y_4 = self.Node_y_4(F.interpolate(y_3_down, size=under.shape[-2:], mode='bilinear'), weights[7]) + y_2

        A_x = self.Node_x_5(x_4, weights[8])
        A_y = self.Node_y_5(y_4, weights[9])

        F_i = A_x * x_1 + A_y * y_1

        return F_i, A_x, A_y

class RetinexModel(nn.Module):
    def __init__(self):
        super(RetinexModel, self).__init__()

        self.Node_div = MixedOp(16, 16)
        self.Node_i_1 = MixedOp(16, 32)
        self.Node_i_2 = MixedOp(32, 16)
        self.Node_l_1 = MixedOp(16, 32)
        self.Node_l_2 = MixedOp(32, 16)
        self.reconstruction = conv_layer(16, 1, 3)
        self.tanh = nn.Tanh()
    
    def forward(self, F_i, weights):
        
        i = self.Node_div(F_i, weights[0])

        l = F_i / (torch.sigmoid(i) + 0.00001)

        i1 = self.Node_i_2(self.Node_i_1(i, weights[1]), weights[2]) + i
        i2 = self.Node_i_2(self.Node_i_1(i1, weights[1]), weights[2]) + i1
        i3 = self.Node_i_2(self.Node_i_1(i2, weights[1]), weights[2]) + i2

        l1 = l / (torch.sigmoid(self.Node_l_2(self.Node_l_1(l, weights[3]), weights[4])) + 0.00001)
        l2 = l1 / (torch.sigmoid(self.Node_l_2(self.Node_l_1(l1, weights[3]), weights[4])) + 0.00001)
        l3 = l2 / (torch.sigmoid(self.Node_l_2(self.Node_l_1(l2, weights[3]), weights[4])) + 0.00001)

        F_f = i3 * l3
        F = self.tanh(self.reconstruction(F_f))

        return F


class Network(nn.Module):

    def __init__(self, args):
        super(Network, self).__init__()
        self.args = args
        
        self.tv = TVLoss()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.grad_loss = GradientLoss()
        self.perc_loss = PerceptualLoss()
        self.color_criterion = LossFunctionlcolor()
        self.psnr_loss = PSNRLoss(max_val=1.)
        self.ssim_loss = SSIMLoss(window_size=5)
        self.mef_ssim = MEFSSIM()
        self.cl = ContrastiveLoss(args)
        
        self.att = AttentionModel()
        self.ret = RetinexModel()

        self._initialize_alphas()
        self._initialize_hypers()

        self.min_num1 = 3
        self.min_num2 = 3
        self.cut_index1 = np.full((10,11), False)
        self.cut_index2 = np.full((5,11), False)

    def forward(self, input):
        o = input[:,0,:,:].unsqueeze(dim=1)
        u = input[:,1,:,:].unsqueeze(dim=1)

        weights_att = F.softmax(self.alphas_att, dim=-1)
        weights_ret = F.softmax(self.alphas_ret, dim=-1)

        f_i_feature, _x1, _x2 = self.att(u, o, weights_att) 
        f = self.ret(f_i_feature, weights_ret)

        return f
    
    def net_named_parameters(self):
        return itertools.chain(self.att.named_parameters(), self.ret.named_parameters())
    
    def net_parameters(self):
        return itertools.chain(self.att.parameters(), self.ret.parameters())

    def _initialize_alphas(self): 
        num_ops = len(SPACE_SIMPLE)

        num_nodes_att = 10
        num_nodes_ret = 5

        self.alphas_att = Variable(1e-2 * torch.ones(num_nodes_att, num_ops).cuda(), requires_grad=True)
        self.alphas_ret = Variable(2e-2 * torch.ones(num_nodes_ret, num_ops).cuda(), requires_grad=True)
        self._arch_parameters = [
            self.alphas_att,
            self.alphas_ret
        ]

    def _initialize_hypers(self): 
        num_loss = 19
        
        self.hypers = Variable(1.25 * torch.ones(num_loss).cuda(), requires_grad=True)
        self._hypers_parameters = [self.hypers]

    def arch_parameters(self):
        return self._arch_parameters

    def hyper_parameters(self):
        return self._hypers_parameters
    
    def calculate_loss(self, x, t):
        target, u, o = t

        weights_hyper = F.softmax(self.hypers, dim=-1)

        aa = self.l1_loss(x, target) 
        a1 = self.l1_loss(x, u) 
        a2 = self.l1_loss(x, o) 
        bb = self.l2_loss(x, target) 
        b1 = self.l2_loss(x, u) 
        b2 = self.l2_loss(x, o) 
        ff = self.grad_loss(x, u, o) 
        gg = self.perc_loss(x, target) 
        g1 = self.perc_loss(x, u) * (-1)
        g2 = self.perc_loss(x, o) * (-1)
        hh = self.ssim_loss(x, target) 
        h1 = self.ssim_loss(x, u) * (-1)
        h2 = self.ssim_loss(x, o) * (-1)
        zz = self.mef_ssim(x, target) * (-1)
        z1 = self.mef_ssim(x, u) * (-1)
        z2 = self.mef_ssim(x, o) * (-1)
        jj = self.color_criterion(x.repeat(1,3,1,1), target.repeat(1,3,1,1)) * 0.15
        mm = self.tv(x)
        dd = self.psnr_loss(x, target) * 0.075

        l = [aa,a1,a2,bb,b1,b2,ff,gg,g1,g2,hh,h1,h2,zz,z1,z2,jj,mm,dd]
        losses = 0.0
        assert len(weights_hyper) == len(l)
        for w, loss in zip(weights_hyper, l):
            losses += w * loss

        return losses, l, weights_hyper

    def calculate_CL(self, x, t):
        target, u, o = t
        loss, ll = self.cl(x, target, u, o)
        loss.requires_grad_()
        return loss, ll
    
    def new(self):
        model_new = Network(self.args).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def new_hyper(self):
        model_new = Network(self.args).cuda()
        for x, y in zip(model_new.hyper_parameters(), self.hyper_parameters()):
            x.data.copy_(y.data)
        return model_new
    
    def genotype_arch(self):

        def _parse(weights):
            max_values = []

            H, W = weights.shape 
            k = W
            for h in range(H):
                values = []
                for w in range(W):
                    values.append((SPACE_SIMPLE[w], weights[h][w]))
                v = sorted(values, key=lambda x:x[1], reverse=True)[:k]
                max_values.append(v)
                
            return max_values

        v_att = _parse(F.softmax(self.alphas_att, dim=-1).data.cpu().numpy())
        v_ret = _parse(F.softmax(self.alphas_ret, dim=-1).data.cpu().numpy())

        Genotype = namedtuple('Genotype', 'v_att  v_ret ')
        
        genotype = Genotype(
            v_att=v_att,
            v_ret=v_ret,
        )

        return genotype
    
    def get_hyper(self):
        weights_hyper = F.softmax(self.hypers, dim=-1).data.cpu().numpy()
        Genotype = namedtuple('hper_Genotype', ' hyper_values ')
        genotype = Genotype(hyper_values=weights_hyper)
        return genotype


    def cut_mins(self, cut_num_1, cut_num_2):
        dim, ops = self.alphas_att.shape
        self.alphas_att.requires_grad = False
        cuts1 = []
        for i in range(dim):
            for _ in range(cut_num_1):
                cnt=0
                min_index = -1
                for j in range(ops):
                    if self.cut_index1[i][j]==False:
                        cnt += 1
                        min_index = j
                if cnt<=self.min_num1:
                    break
                for j in range(ops):
                    if self.cut_index1[i][j]==False and self.alphas_att[i][j]<self.alphas_att[i][min_index]:
                        min_index = j
                self.cut_index1[i][min_index] = True
                cuts1.append((i, min_index))
                
        for i in range(dim):
            for j in range(ops):
                if self.cut_index1[i][j]:
                    self.alphas_att[i][j] = -99
        self.alphas_att.requires_grad = True
        

        dim, ops = self.alphas_ret.shape
        self.alphas_ret.requires_grad = False
        cuts2 = []
        for i in range(dim):
            for _ in range(cut_num_2):
                cnt=0
                min_index = -1
                for j in range(ops):
                    if self.cut_index2[i][j]==False:
                        cnt += 1
                        min_index = j
                if cnt<=self.min_num2:
                    break
                for j in range(ops):
                    if self.cut_index2[i][j]==False and self.alphas_ret[i][j]<self.alphas_ret[i][min_index]:
                        min_index = j
                self.cut_index2[i][min_index] = True
                cuts2.append((i, min_index))

        for i in range(dim):
            for j in range(ops):
                if self.cut_index2[i][j]:
                    self.alphas_ret[i][j] = -99
        self.alphas_ret.requires_grad = True

        return cuts1, cuts2

    def cut(self):
        self.alphas_att.requires_grad = False
        dim, ops = self.alphas_att.shape
        for i in range(dim):
            for j in range(ops):
                if self.cut_index1[i][j]:
                    self.alphas_att[i][j] = -99
        self.alphas_att.requires_grad = True

        self.alphas_ret.requires_grad = False
        dim, ops = self.alphas_ret.shape
        for i in range(dim):
            for j in range(ops):
                if self.cut_index2[i][j]:
                    self.alphas_ret[i][j] = -99
        self.alphas_ret.requires_grad = True
        