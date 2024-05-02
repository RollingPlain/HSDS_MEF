import os
import cv2
import torch
import utils
import argparse
import torch.utils
import torchvision
import torch.backends.cudnn as cudnn
from model.model_test import Network
from tqdm import tqdm

parser = argparse.ArgumentParser("ruas")
parser.add_argument('--under', type=str, default='./Data/test/under')
parser.add_argument('--over', type=str, default='./Data/test/over')
parser.add_argument('--result_path', type=str, default='./Data/test/result')
parser.add_argument('--checkpoint', type=str, default='./ckp/final.pt')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cudnn.benchmark = True
cudnn.enabled = True

assert set(os.listdir(args.under))==set(os.listdir(args.over))
if not os.path.exists(args.result_path):
    os.makedirs(args.result_path)

model = Network()
model = model.cuda()

model.load_state_dict(torch.load(args.checkpoint))

model.eval()

with torch.no_grad():
    for img in tqdm(os.listdir(args.over), ascii='>='):
        o = cv2.imread(os.path.join(args.over, img))
        u = cv2.imread(os.path.join(args.under, img))

        o = torchvision.transforms.ToTensor()(cv2.cvtColor(o, cv2.COLOR_BGR2YCrCb))
        u = torchvision.transforms.ToTensor()(cv2.cvtColor(u, cv2.COLOR_BGR2YCrCb))
        o_y = o[[0], :, :].unsqueeze(dim=0)
        o_cr = o[[1], :, :].unsqueeze(dim=0)
        o_cb = o[[2], :, :].unsqueeze(dim=0)

        u_y = u[[0], :, :].unsqueeze(dim=0)
        u_cr = u[[1], :, :].unsqueeze(dim=0)
        u_cb = u[[2], :, :].unsqueeze(dim=0)

        o_y, o_cr, o_cb, u_y, u_cr, u_cb = utils.togpu_6('cuda:0', o_y, o_cr, o_cb, u_y, u_cr, u_cb)

        output = model(torch.cat((o_y, u_y), dim=1))

        cr, cb = utils.Get_color_MEF_tensor(u_cr, u_cb, o_cr, o_cb)

        output = utils.YCrCb2RGB(torch.cat((output, cr, cb), dim=1))

        torchvision.utils.save_image(output, os.path.join(args.result_path, img))

