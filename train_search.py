import os
import sys
import math
import time
import utils
import torch
import logging
import argparse
import torch.utils
import torchvision
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from dataset.dataset import Data_fusion
from model.model_search import Network
from arch.architect import Architect
from arch.Hyper_optimizer import HyperOptimizer

parser = argparse.ArgumentParser("ruas")
parser.add_argument('--train_data', type=str, default='./Data/train')
parser.add_argument('--natural_path', type=str, default='./Data/train/CoCo_val2017')
parser.add_argument('--crop_size', type=tuple, default=(256, 256))
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=1, help='report frequency')
parser.add_argument('--warmup', type=int, default=-1)
parser.add_argument('--cut_freq', type=int, default=1)

parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--save', type=str, default='', help='path to save the model')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--unrolled', type=bool, default=True, help='use one-step unrolled validation loss')

parser.add_argument('--arch_learning_rate', type=float, default=0.2, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

parser.add_argument('--hyper_learning_rate', type=float, default=0.1, help='learning rate for hyper encoding')
parser.add_argument('--hyper_weight_decay', type=float, default=1e-3, help='weight decay for hyper encoding')

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
args.save = 'Search{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
EXP_path = args.save
if not os.path.isdir(EXP_path):
    os.mkdir(EXP_path)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(EXP_path, 'S_EXP.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)

    model = Network(args=args)
    model = model.cuda()

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    architect = Architect(model, args)
    hyper_architect = HyperOptimizer(model, args)

    train_dataset = Data_fusion(root_dir=args.train_data, crop_size=args.crop_size)
    val_dataset = Data_fusion(root_dir=args.train_data, crop_size=args.crop_size)

    train_data_size = len(train_dataset)
    indices = list(range(train_data_size))
    split = int(np.floor(0.5 * train_data_size))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
                                              pin_memory=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:train_data_size]),
                                              pin_memory=True, num_workers=0, drop_last=True)

    optimizer = torch.optim.Adam(model.net_parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)


    total_step = 0
    for epoch in range(args.epochs):
        total_search_loss = 0
        total_contrast_loss = 0
        lr = scheduler.get_lr()[0]

        if epoch<=args.warmup:
            logging.info('----------- warmup {} -----------'.format(args.warmup))
        logging.info("-------------- epoch {} lr {} --------------".format(epoch, lr))
        model.train()
        for step, data in enumerate(train_loader):
            names, o_y, o_cr, o_cb, u_y, u_cr, u_cb, g_y, g_cr, g_cb = data.values()
            o_y, o_cr, o_cb, u_y, u_cr, u_cb, g_y, g_cr, g_cb = utils.togpu_9('cuda:0', o_y, o_cr, o_cb, u_y, u_cr, u_cb, g_y, g_cr, g_cb)
            input = torch.cat((o_y, u_y), dim=1)
            target = [g_y.clone(), u_y.clone(), o_y.clone()]

            if epoch>args.warmup:
                _, o_y_arch, _, _, u_y_arch, _, _, g_y_arch, _, _ = next(iter(val_loader)).values()
                o_y_arch, u_y_arch, g_y_arch = utils.togpu_3('cuda:0', o_y_arch, u_y_arch, g_y_arch)
                input_arch = torch.cat((o_y_arch, u_y_arch), dim=1)
                target_arch = [g_y_arch, u_y_arch, o_y_arch]

                _, o_y_hyper, _, _, u_y_hyper, _, _, g_y_hyper, _, _ = next(iter(val_loader)).values()
                o_y_hyper, u_y_hyper, g_y_hyper = utils.togpu_3('cuda:0', o_y_hyper, u_y_hyper, g_y_hyper)
                input_hyper = torch.cat((o_y_hyper, u_y_hyper), dim=1)
                target_hyper = [g_y_hyper, u_y_hyper, o_y_hyper]

                architect.step(input, target, input_arch, target_arch, lr, optimizer, args.unrolled)
                model.cut() 

            output = model(input)

            loss_search, _, _ = model.calculate_loss(output, target)
            contrast_loss, _ = model.calculate_CL(output, target)
            loss = loss_search

            if not math.isfinite(loss.item()):
                logging.info("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch>args.warmup:
                hyper_architect.step(input, target, input_hyper, target_hyper, lr, optimizer, args.unrolled)

            total_step += 1
            total_search_loss += loss_search.item()
            total_contrast_loss += contrast_loss.item()
            logging.info('step:{}, total_step:{}, search_loss:{}, CT_loss:{}'.format(step, total_step, loss_search.item(), contrast_loss.item()))

            if total_step % args.report_freq == 0:
                logging.info('step:{}, total_step:{}, search_loss:{}, CT_loss:{}'.format(step, total_step, loss_search.item(), contrast_loss.item()))
                cr, cb = utils.Get_color_MEF_tensor(u_cr, u_cb, o_cr, o_cb) 
                output_1 = torch.cat((output, cr, cb), dim=1)
                output_1 = utils.YCrCb2RGB(output_1)
                input_o = utils.YCrCb2RGB(torch.cat((o_y, o_cr, o_cb), dim=1))
                input_u = utils.YCrCb2RGB(torch.cat((u_y, u_cr, u_cb), dim=1))
                input_g = utils.YCrCb2RGB(torch.cat((g_y, g_cr, g_cb), dim=1))
                print(names)

                torchvision.utils.save_image(input_o, os.path.join(EXP_path, 'over.png'))
                torchvision.utils.save_image(input_u, os.path.join(EXP_path, 'under.png'))
                torchvision.utils.save_image(input_g, os.path.join(EXP_path, 'GT.png'))
                torchvision.utils.save_image(output, os.path.join(EXP_path, 'output.png'))
                torchvision.utils.save_image(output_1, os.path.join(EXP_path, 'output_color.png'))

                gno(model, epoch, lr, step, total_step)

        logging.info('epoch:{}, total_search_loss:{}, total_contrast_loss:{}'.format(epoch, total_search_loss, total_contrast_loss))

        scheduler.step()
        if epoch % args.cut_freq == 0:
            cuts1, cuts2 = model.cut_mins(cut_num_1=2, cut_num_2=2) 
            logging.info('cutting att {}'.format(cuts1))
            logging.info('cutting ret {}'.format(cuts2))
            gno(model, epoch, lr, step, total_step)


def gno(model, epoch, lr, step, total_step):
    logging.info('------------ epoch {} lr {} step {} total_step {}----------------'.format(epoch, lr, step, total_step))
    logging.info('α of arch:')
    genotype = model.genotype_arch()
    logging.info('arch = %s', genotype)
    logging.info('v_att = %s', genotype.v_att)
    logging.info('v_ret = %s', genotype.v_ret)

    logging.info('β of hyper:')
    genotype = model.get_hyper()
    logging.info('hyper = %s', genotype)
    logging.info('----------------------------------------------')



if __name__ == '__main__':
    main()
