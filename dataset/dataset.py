import cv2
import torch
import os.path
from torchvision import transforms
from torch.utils.data import Dataset

class Data_fusion(Dataset):
    def __init__(self, root_dir, crop_size=(256, 256)):
        self.root_dir = root_dir
        self.o_path = os.path.join(self.root_dir, 'over')
        self.u_path = os.path.join(self.root_dir, 'under')
        self.g_path = os.path.join(self.root_dir, 'ref')

        assert len(os.listdir(self.o_path)) == len(os.listdir(self.u_path)) 

        self.transform = transforms.Compose([transforms.RandomResizedCrop(crop_size), transforms.RandomHorizontalFlip(p=0.5)])
        self.totensor = transforms.Compose([transforms.ToTensor()])

        self.img_list = os.listdir(self.o_path)

    def __getitem__(self, idx):
        seed = torch.random.seed()

        name = self.img_list[idx]

        o_item_path = os.path.join(self.o_path, name)
        u_item_path = os.path.join(self.u_path, name)
        g_item_path = os.path.join(self.g_path, name)
        o = cv2.imread(o_item_path)
        u = cv2.imread(u_item_path)
        g = cv2.imread(g_item_path)
        o = self.trans(self.totensor(cv2.cvtColor(o, cv2.COLOR_BGR2YCrCb)), seed)
        u = self.trans(self.totensor(cv2.cvtColor(u, cv2.COLOR_BGR2YCrCb)), seed)
        g = self.trans(self.totensor(cv2.cvtColor(g, cv2.COLOR_BGR2YCrCb)), seed) # CHW
        o_y = o[0, :, :].unsqueeze(dim=0)
        o_cr = o[1, :, :].unsqueeze(dim=0)
        o_cb = o[2, :, :].unsqueeze(dim=0)

        u_y = u[0, :, :].unsqueeze(dim=0)
        u_cr = u[1, :, :].unsqueeze(dim=0)
        u_cb = u[2, :, :].unsqueeze(dim=0)

        g_y = g[0, :, :].unsqueeze(dim=0)
        g_cr = g[1, :, :].unsqueeze(dim=0)
        g_cb = g[2, :, :].unsqueeze(dim=0)

        result = {'name':name.split('.')[0],
                  'o_y':o_y, 'o_cr':o_cr, 'o_cb':o_cb,
                  'u_y':u_y, 'u_cr':u_cr, 'u_cb':u_cb,
                  'g_y':g_y, 'g_cr':g_cr, 'g_cb':g_cb,
                  }

        return result

    def trans(self, x, seed):
        torch.random.manual_seed(seed)
        x = self.transform(x)
        return x

    def __len__(self):
        return len(self.img_list)
