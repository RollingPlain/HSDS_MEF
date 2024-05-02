import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn
from .val_dataset import Data_CT
from torch.utils.data import DataLoader


class PSNR(nn.Module):
    def __init__(self):
        super(PSNR, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, img1, img2):
        m = self.mse(img1, img2)
        if m.item() < 1e-10:
            return 100
        psnr = 20 * math.log10(1 / math.sqrt(m))
        return psnr

class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

class LossFunctionlcolor(nn.Module):
    def __init__(self):
        super(LossFunctionlcolor, self).__init__()
    def forward(self, image, label):
        img1 = torch.reshape(image, [3, -1])
        img2 = torch.reshape(label, [3, -1])
        clip_value = 0.999999
        norm_vec1 = torch.nn.functional.normalize(img1, p=2, dim=0)
        norm_vec2 = torch.nn.functional.normalize(img2, p=2, dim=0)
        temp = norm_vec1 * norm_vec2
        dot = temp.sum(dim=0)
        dot = torch.clamp(dot, -clip_value, clip_value)
        angle = torch.acos(dot) * (180 / math.pi)
        return 0.1*torch.mean(angle)

class GradientLoss(nn.Module):
    # 梯度loss
    def __init__(self):
        super(GradientLoss, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
        self.L1loss = nn.L1Loss()

    def forward(self, x, s1, s2):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        grad_f = torch.abs(sobelx) + torch.abs(sobely)

        sobelx_s1 = F.conv2d(s1, self.weightx, padding=1)
        sobely_s1 = F.conv2d(s1, self.weighty, padding=1)
        grad_s1 = torch.abs(sobelx_s1) + torch.abs(sobely_s1)

        sobelx_s2 = F.conv2d(s2, self.weightx, padding=1)
        sobely_s2 = F.conv2d(s2, self.weighty, padding=1)
        grad_s2 = torch.abs(sobelx_s2) + torch.abs(sobely_s2)

        grad_max = torch.max(grad_s1, grad_s2)

        loss = self.L1loss(grad_f, grad_max)
        return loss

class FeatureHook:
    def __init__(self, module):
        self.features = None
        self.hook = module.register_forward_hook(self.on)

    def on(self, module, inputs, outputs):
        self.features = outputs

    def close(self):
        self.hook.remove()

class PerceptualLoss(nn.Module):
    def __init__(self, blocks=[0,1,2,3]):
        super().__init__()
        self.feature_loss = nn.MSELoss()

        #VGG16 contains 5 blocks - 3 convolutions per block and 3 dense layers towards the end

        vgg = vgg16_bn(pretrained=True).features
        vgg.eval()

        for param in vgg.parameters():
            param.requires_grad = False

        bns = [i - 2 for i, m in enumerate(vgg) if isinstance(m, nn.MaxPool2d)]
        assert all(isinstance(vgg[bn], nn.BatchNorm2d) for bn in bns)

        self.hooks = [FeatureHook(vgg[bns[i]]) for i in blocks]
        self.features = vgg[0: bns[blocks[-1]] + 1]

    def forward(self, inputs, targets):

        # normalize foreground pixels to ImageNet statistics for pre-trained VGG
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        
        B, C, H, W = inputs.shape
        if C == 1:
            inputs = inputs.repeat(1, 3, 1, 1)
            targets = targets.repeat(1, 3, 1, 1)
        inputs = F.normalize(inputs, dim=1)
        targets = F.normalize(targets, dim=1)

        # extract feature maps
        self.features(inputs)
        input_features = [hook.features.clone() for hook in self.hooks]

        self.features(targets)
        target_features = [hook.features for hook in self.hooks]

        loss = 0.0
        
        for lhs, rhs in zip(input_features, target_features):
            lhs = lhs.view(lhs.size(0), -1)
            rhs = rhs.view(rhs.size(0), -1)
            loss += self.feature_loss(lhs, rhs)

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.vgg_loss = PerceptualLoss_CR()
        self.vgg_loss.cuda()

        val_data = Data_CT(root_dir=args.natural_path, crop_size=args.crop_size)
        self.val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True,
                                              pin_memory=True, drop_last=True)

        over_data = Data_CT(root_dir=os.path.join(args.train_data, 'over'), crop_size=args.crop_size)
        self.over_loader = DataLoader(dataset=over_data, batch_size=args.batch_size, shuffle=True,
                                              pin_memory=True, drop_last=True)
        
        under_data = Data_CT(root_dir=os.path.join(args.train_data, 'under'), crop_size=args.crop_size)
        self.under_loader = DataLoader(dataset=under_data, batch_size=args.batch_size, shuffle=True,
                                              pin_memory=True, drop_last=True)
        
        self.extra_num = 3

    def forward(self, x, targets, s1, s2):
        fake_C = x
        gt = targets
        under = s1
        over = s2

        self.vgg_loss.compute_fake_C(fake_C)

        loss_DH3 = 0
        loss_DH4 = 0

        y_val = None
        
        for _ in range(self.extra_num):
            n0, y_val = next(iter(self.val_loader)).values()
            y_val = y_val.cuda()
            n1, o_val = next(iter(self.over_loader)).values() 
            o_val = o_val.cuda()
            n2, u_val = next(iter(self.under_loader)).values() 
            u_val = u_val.cuda()
            loss_natural = self.vgg_loss.compute_vgg_loss(y_val)
            loss_DH3 += loss_natural / (self.vgg_loss.compute_vgg_loss(u_val) + 1e-7)

            loss_DH4 += loss_natural / (self.vgg_loss.compute_vgg_loss(o_val) + 1e-7)

            del y_val
            del o_val
            del u_val

        loss_vgg_b = self.vgg_loss.compute_vgg_loss(gt)

        loss_DH1 =  loss_vgg_b / (self.vgg_loss.compute_vgg_loss(under) + 1e-7)
        
        loss_DH2 = loss_vgg_b / (self.vgg_loss.compute_vgg_loss(over) + 1e-7)
        
        loss_G = 4 * loss_vgg_b + loss_DH1 + loss_DH2 + loss_DH3 + loss_DH4

        return loss_G, [loss_natural.item(), loss_vgg_b.item(), 0, 0]


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        # self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, X):
        h = F.relu(self.conv1_1(X), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        x0 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        x1 = h
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        conv4_3 = self.conv4_3(h)
        h = F.relu(conv4_3, inplace=True)

        relu5_1 = F.relu(self.conv5_1(h), inplace=True)

        return relu5_1, x1, x0


def vgg_preprocess(batch):
    batch = batch.repeat(1, 3, 1, 1)
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    return batch


class PerceptualLoss_CR(nn.Module):
    def __init__(self):
        super(PerceptualLoss_CR, self).__init__()
        self.instancenorm1 = nn.InstanceNorm2d(512, affine=False)
        self.instancenorm2 = nn.InstanceNorm2d(256, affine=False)
        self.instancenorm3 = nn.InstanceNorm2d(128, affine=False)

        self.vgg = load_vgg16()
        self.vgg.eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
        
    def compute_fake_C(self, img):
        img_vgg = vgg_preprocess(img)
        self.img_fea1, self.img_fea2, self.img_fea3 = self.vgg(img_vgg)

    def compute_vgg_loss(self, target):
        target_vgg = vgg_preprocess(target)
        target_fea1, target_fea2, target_fea3 = self.vgg(target_vgg)
        img_fea1 = self.img_fea1
        img_fea2 = self.img_fea2
        img_fea3 = self.img_fea3


        loss = torch.mean((self.instancenorm1(target_fea1) - self.instancenorm1(img_fea1)) ** 2) +\
              torch.mean((self.instancenorm2(target_fea2) - self.instancenorm2(img_fea2)) ** 2) +\
              torch.mean((self.instancenorm3(target_fea3) - self.instancenorm3(img_fea3)) ** 2)
        return loss

def load_vgg16():
    vgg = Vgg16()
    vgg.load_state_dict(torch.load('./ckp/vgg16.weight'), strict=False)
    return vgg
