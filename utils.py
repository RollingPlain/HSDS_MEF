import torch
import numpy as np
import torchvision.transforms as transforms


def Get_color_MEF_tensor(cb1, cr1, cb2, cr2): # input cr1, cb1, cr2, cb2

      cb1s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cb1, 1, dim=0)]
      cr1s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cr1, 1, dim=0)]
      cb2s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cb2, 1, dim=0)]
      cr2s = [x.clone().squeeze().cpu().numpy()*255 for x in torch.split(cr2, 1, dim=0)]
      cb = []
      cr = []
      for Cb1, Cr1, Cb2, Cr2 in zip(cb1s, cr1s, cb2s, cr2s):
        H, W = Cb1.shape
        Cb = np.ones((H, W))
        Cr = np.ones((H, W))
        for k in range(H):
            for n in range(W):
                if abs(Cb1[k, n] - 128) == 0 and abs(Cb2[k, n] - 128) == 0:
                    Cb[k, n] = 128
                else:
                    middle_1 = Cb1[k, n] * abs(Cb1[k, n] - 128) + Cb2[k, n] * abs(Cb2[k, n] - 128)
                    middle_2 = abs(Cb1[k, n] - 128) + abs(Cb2[k, n] - 128)
                    Cb[k, n] = middle_1 / middle_2
                if abs(Cr1[k, n] - 128) == 0 and abs(Cr2[k, n] - 128) == 0:
                    Cr[k, n] = 128
                else:
                    middle_3 = Cr1[k, n] * abs(Cr1[k, n] - 128) + Cr2[k, n] * abs(Cr2[k, n] - 128)
                    middle_4 = abs(Cr1[k, n] - 128) + abs(Cr2[k, n] - 128)
                    Cr[k, n] = middle_3 / middle_4
        cb.append(transforms.ToTensor()(Cb.astype(np.uint8)).reshape(1, 1, H, W))
        cr.append(transforms.ToTensor()(Cr.astype(np.uint8)).reshape(1, 1, H, W))
      cb = torch.cat(cb, dim=0).cuda()
      cr = torch.cat(cr, dim=0).cuda()
      return cb, cr
      
            
def togpu_9(device, x1, x2, x3, x4, x5, x6, x7, x8, x9):
   x1 = x1.to(device)
   x2 = x2.to(device)
   x3 = x3.to(device)
   x4 = x4.to(device)
   x5 = x5.to(device)
   x6 = x6.to(device)
   x7 = x7.to(device)
   x8 = x8.to(device)
   x9 = x9.to(device)
   return x1, x2, x3, x4, x5, x6, x7, x8, x9


def togpu_6(device, x1, x2, x3, x4, x5, x6):
    x1 = x1.to(device)
    x2 = x2.to(device)
    x3 = x3.to(device)
    x4 = x4.to(device)
    x5 = x5.to(device)
    x6 = x6.to(device)
    return  x1, x2, x3, x4, x5, x6


def togpu_3(device, x1, x2, x3):
    x1 = x1.to(device)
    x2 = x2.to(device)
    x3 = x3.to(device)
    return  x1, x2, x3

def YCrCb2RGB(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
            .transpose(1, 3)
            .transpose(2, 3)
    )
    return out


def RGB2YCrCb(input_im):
    im_flat = input_im.transpose(1, 3).transpose(
        1, 2).reshape(-1, 3)  # (nhw,c)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5
    Y = torch.unsqueeze(Y, 1)
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (
        temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,
        )
            .transpose(1, 3)
            .transpose(2, 3)
    )
    return out


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6
