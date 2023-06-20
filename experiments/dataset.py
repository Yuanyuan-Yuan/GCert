import os
import json
import random
from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


class CelebARecog(Dataset):
    def __init__(self, 
                img_dir='/YOUR_DATA_DIR/celeba_crop128/',
                name2id_path='/YOUR_DATA_DIR/CelebA_name_to_ID.json',
                id2name_path='/YOUR_DATA_DIR/CelebA_ID_to_name.json',
                split='train',
                num_tuple=800,
                img_size=64):
        super(CelebARecog).__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transforms.Compose([
                       transforms.Resize(img_size),
                       transforms.CenterCrop(img_size),
                       transforms.ToTensor()
               ])
        with open(name2id_path, 'r') as f:
            self.name2id = json.load(f)
        with open(id2name_path, 'r') as f:
            self.id2name = json.load(f)
        self.name_dir = img_dir + split + '/'
        self.name_list = sorted(os.listdir(self.name_dir))
        self.pos_pair_list = []
        self.neg_pair_list = []
        for i in tqdm(range(min(num_tuple, len(self.name_list)))):
            name = self.name_list[i]
            ID = self.name2id[name]
            if (len(self.id2name[ID]) == 1) or (len([x for x in self.id2name[ID] if x in self.name_list]) == 0):
                continue
            same = random.choice(self.id2name[ID])
            while (same not in self.name_list) or (same == name):
                same = random.choice(self.id2name[ID])
            diff = random.choice(self.name_list)        
            while diff in self.id2name[ID]:
                diff = random.choice(self.name_list)
            self.pos_pair_list.append((name, same))
            self.neg_pair_list.append((name, diff))

        print('Total %d' % len(self.pos_pair_list))

    def __len__(self):
        return len(self.pos_pair_list)

    def __getitem__(self, idx):
        (name, same) = self.pos_pair_list[idx]
        (_, diff) = self.neg_pair_list[idx]
        img = Image.open(self.name_dir + name)
        img = self.transform(img)
        img_same = Image.open(self.name_dir + same)
        img_same = self.transform(img_same)
        img_diff = Image.open(self.name_dir + diff)
        img_diff = self.transform(img_diff)
        return img, img_same, img_diff

class CelebAGen(Dataset):
    def __init__(self, 
                img_dir='/YOUR_DATA_DIR/celeba_crop128/',
                split='train',
                num_img=200000,
                img_size=64,
                gray=False):
        super(CelebAGen).__init__()
        assert split in ['train', 'val', 'test']
        self.transform = transforms.Compose([
                       transforms.Grayscale(num_output_channels=1),
                       transforms.Resize(img_size),
                       transforms.CenterCrop(img_size),
                       transforms.ToTensor()
               ]) if gray else transforms.Compose([
                       transforms.Resize(img_size),
                       transforms.CenterCrop(img_size),
                       transforms.ToTensor()
               ])
        self.name_dir = img_dir + split + '/'
        self.name_list = sorted(os.listdir(self.name_dir))
        self.name_list = self.name_list[:num_img]

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        name = self.name_list[idx]
        img = Image.open(self.name_dir + name)
        img = self.transform(img)
        return img