import cv2
import torch
import os.path
from torchvision import transforms
from torch.utils.data import Dataset

class Data_CT(Dataset):
    def __init__(self, root_dir, crop_size=(256, 256)):
        self.root_dir = root_dir
        self.img_list = os.listdir(self.root_dir)

        self.transform = transforms.Compose([transforms.RandomResizedCrop(crop_size), transforms.RandomHorizontalFlip(p=0.5)])

        self.totensor = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        seed = torch.random.seed()

        name = self.img_list[idx]

        o_item_path = os.path.join(self.root_dir, name)
        o = cv2.imread(o_item_path)
        o = self.trans(self.totensor(cv2.cvtColor(o, cv2.COLOR_BGR2YCrCb)), seed)
        o_y = o[0, :, :].unsqueeze(dim=0)

        result = {'name':name.split('.')[0], 'o_y':o_y}

        return result

    def trans(self, x, seed):
        torch.random.manual_seed(seed)
        x = self.transform(x)
        return x

    def __len__(self):
        return len(self.img_list)
