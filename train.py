import os
import sys
import math
import time
import torch
import utils
import logging
import argparse
import torch.utils
import torchvision
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from dataset.dataset import Data_fusion
from model.model_train import Network

parser = argparse.ArgumentParser("ruas")
parser.add_argument('--train_data', type=str, default='./Data/train') 
parser.add_argument('--natural_path', type=str, default='./Data/train/CoCo_val2017')
parser.add_argument('--crop_size', type=tuple, default=(256, 256))
parser.add_argument('--batch_size', type=int, default=2, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.006, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=1e-9, help='min learning rate')
parser.add_argument('--report_freq', type=float, default=3, help='report frequency')
parser.add_argument('--test_freq', type=float, default=4, help='report frequency')
parser.add_argument('--epochs', type=int, default=8, help='num of training epochs')
parser.add_argument('--save', type=str, default='', help='path to save the model')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
args.save = 'Train{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
EXP_path = args.save
if not os.path.isdir(EXP_path):
    os.mkdir(EXP_path)
model_path = EXP_path + '/model/'
if not os.path.isdir(model_path):
    os.mkdir(model_path)



log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(EXP_path, 'T_EXP.log'))
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

    train_dataset = Data_fusion(root_dir=args.train_data, crop_size=args.crop_size)

    train_data_size = len(train_dataset)
    indices = list(range(train_data_size))
    logging.info('dataset size:{}'.format(train_data_size))

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,sampler=torch.utils.data.sampler.SubsetRandomSampler(indices),
                                              pin_memory=True, num_workers=0, drop_last=True)

    optimizer = torch.optim.Adam(model.net_parameters(), lr=args.learning_rate)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs)*len(train_loader), eta_min=args.learning_rate_min)

    total_step = 0
    for epoch in range(1, args.epochs+1):
        total_search_loss = 0
        total_contrast_loss = 0
        lr = scheduler.get_lr()[0]


        logging.info("-------------- epoch {} lr {} --------------".format(epoch, lr))
        model.train()
        for step, data in enumerate(train_loader):
            names, o_y, o_cr, o_cb, u_y, u_cr, u_cb, g_y, g_cr, g_cb = data.values()
            o_y, o_cr, o_cb, u_y, u_cr, u_cb, g_y, g_cr, g_cb = utils.togpu_9('cuda:0', o_y, o_cr, o_cb, u_y, u_cr, u_cb, g_y, g_cr, g_cb)
            input = torch.cat((o_y, u_y), dim=1)
            target = [g_y.clone(), u_y.clone(), o_y.clone()]

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


            total_step += 1
            total_search_loss += loss_search.item()
            total_contrast_loss += contrast_loss.item()

            logging.info('lr:{}, step:{}, total_step:{}, search_loss:{}, contrast_loss:{}'.format(scheduler.get_lr()[0], step, total_step, loss_search.item(), contrast_loss.item()))
            scheduler.step()

            if total_step % args.report_freq == 0:
                logging.info('step:{}, total_step:{}, search_loss:{}, contrast_loss:{}'.format(step, total_step, loss_search.item(), contrast_loss.item()))
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
                torchvision.utils.save_image(g_y, os.path.join(EXP_path, 'gt_y.png'))
                torchvision.utils.save_image(output, os.path.join(EXP_path, 'output.png'))
                torchvision.utils.save_image(output_1, os.path.join(EXP_path, 'output_color.png'))

        logging.info('epoch:{}, total_search_loss:{}, total_contrast_loss:{}'.format(epoch, total_search_loss, total_contrast_loss))

        if epoch % args.test_freq == 0 and epoch!=0:
            state_dict_model = {k:v for k, v in model.state_dict().items() if 'att' in k or 'ret' in k}
            torch.save(state_dict_model, os.path.join(model_path, '{}.pt'.format(epoch)))


if __name__ == '__main__':
    main()
