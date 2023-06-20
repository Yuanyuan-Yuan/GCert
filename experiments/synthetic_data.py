############################################################
# This script implements the synthetic dataset class for   #
# our ablation study. You can use the SyntheticDataset as  #
# a torchvision dataset class.                             #
############################################################

import os
import json
import cv2
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def synthesize(
        use_translate=False,
        use_scale=False,
        use_rotate=False,
        use_color=False,
        save_path=None):
    start = (8, 8) # (0, 0) --> (16, 16)
    shape = (16, 16) # (4, 4) --> (16, 16)
    if use_color:
        img = np.zeros((32, 32, 3), np.uint8)
        B = int(random.randint(0, 1) * 255)
        G = int(random.randint(0, 1) * 255)
        R = int(random.randint(0, 1) * 255)
        color = (B, G, R)
    else:
        img = np.zeros((32, 32, 1), np.uint8)
        color = (255)

    (dx, dy) = (0, 0)
    (sx, sy) = (1, 1)
    if use_translate:
        dx = random.randint(0, 8) * (1 if random.randint(0, 1) > 0 else -1)
        dy = random.randint(0, 8) * (1 if random.randint(0, 1) > 0 else -1)
    if use_scale:
        sx = random.randint(5, 15) / 10
        sy = random.randint(5, 15) / 10

    # print((start[0]+dx, start[1]+dy))
    # print((start[0]+dx+shape[0]*sx, start[1]+dy+shape[1]*sy))
    # print((B, G, R))

    (x1, y1) = (int(start[0] + dx), int(start[1] + dy))
    cv2.rectangle(
        img,
        pt1=(x1, y1),
        pt2=(int(x1 + shape[0] * sx), int(y1 + shape[1] * sy)),
        color=color,
        thickness=-1
    )
    
    if use_rotate:
        (cols, rows) = (32, 32)
        R = random.randint(-9, 9) * 10
        M = cv2.getRotationMatrix2D((cols/2, rows/2), R, 1)
        img = cv2.warpAffine(img, M, (cols, rows))
    
    if save_path is not None:
        cv2.imwrite(save_path, img)
    return img if use_color else img[:, :, 0]


def to_PIL(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img)
    return im_pil


class SyntheticDataset(Dataset):
    def __init__(self, n_data=50000):
        super(SyntheticDataset).__init__()
        self.n_data = n_data
        self.transform = transforms.Compose([
                    transforms.ToTensor(), 
                ])

    def __len__(self):
        return self.n_data

    def __getitem__(self, idx):
        image_pil = to_PIL(synthesize(
                use_translate=True,
                use_scale=True,
                use_rotate=False,
                use_color=False
            ))
        image_tensor = self.transform(image_pil)
        # print(image_tensor.size())
        return image_tensor