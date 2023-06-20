import os
import json
import cv2
import numpy as np

import imgaug.augmenters as iaa

import torch
import torch.nn as n
import torch.nn.functional as F

import tool

__all__ = [
    'Noise', 'Brightness', 'Contrast', 'Blur',
    'Translation', 'Scale', 'Rotation', 'Shear', 'Reflection',
    'Cloud', 'Fog', 'Snow', 'Rain',
]

class Transformation:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.init_id()

    def torch2cv(self, tsr):
        assert tsr.size(0) == 1
        # tsr.max() == 1 and tsr.min() == -1
        tsr = tool.general_normalize_inv(tsr)
        arr = tsr[0].detach().transpose(0, 2).cpu().numpy() * 255
        return arr

    def cv2torch(self, arr):
        arr = arr.astype(np.float32)
        tsr = torch.from_numpy(arr / 255).transpose(0, 2).unsqueeze(0)
        tsr = tool.general_normalize(tsr)#.to(self.device)
        return tsr

    def cv_pad(self, src, dst):
        H_src, W_src, C = src.shape
        H_dst, W_dst = dst.shape[:2]
        
        top = np.max((H_src - H_dst) // 2, 0)
        left = np.max((W_src - W_dst) // 2, 0)
        
        if len(dst.shape) == 2:
            COLOR = 0
            padded = np.full((H_src, W_src, C), COLOR, dtype=np.uint8)
            padded[top:top+H_dst, left:left+W_dst, 0] = dst
        else:
            COLOR = (0, 0, 0)
            padded = np.full((H_src, W_src, C), COLOR, dtype=np.uint8)
            padded[top:top+H_dst, left:left+W_dst] = dst
        return padded

    def init_id(self):
        raise NotImplementedError

    def mutate(self, x):
        raise NotImplementedError

    def extent(self):
        raise NotImplementedError


class Noise(Transformation):
    def init_id(self):
        self.category = 'pixel'
        self.name = 'noise'

    def mutate(self, seed):
        x = seed['x']
        x_ = x + self.extent() * torch.randn(x.size())#.to(self.device)
        return x_, seed['z']

    def extent(self):
        return np.random.uniform(0, 1)

class Brightness(Transformation):
    def init_id(self):
        self.category = 'pixel'
        self.name = 'brightness'

    def mutate(self, seed):
        x = seed['x']
        arr = self.torch2cv(x)
        x_ = cv2.convertScaleAbs(arr, beta=self.extent(), alpha=1)
        return self.cv2torch(x_), seed['z']

    def extent(self):
        return 10 + 10 * np.random.randint(7)

class Contrast(Transformation):
    def init_id(self):
        self.category = 'pixel'
        self.name = 'contrast'

    def mutate(self, seed):
        x = seed['x']
        arr = self.torch2cv(x)
        x_ = cv2.convertScaleAbs(arr, beta=0, alpha=self.extent())
        return self.cv2torch(x_), seed['z']

    def extent(self):
        return 0.8 + 0.2 * np.random.randint(7)

class Blur(Transformation):
    def init_id(self):
        self.category = 'pixel'
        self.name = 'blur'

    def mutate(self, seed):
        x = seed['x']
        arr = self.torch2cv(x)
        x_ = self.extent()(arr)
        return self.cv2torch(x_), seed['z']

    def extent(self):
        blr = np.random.choice([
            lambda img: cv2.blur(img, (3, 3)),
            lambda img: cv2.blur(img, (4, 4)),
            lambda img: cv2.blur(img, (5, 5)),
            lambda img: cv2.blur(img, (6, 6)),
            lambda img: cv2.GaussianBlur(img, (3, 3), 0),
            lambda img: cv2.GaussianBlur(img, (5, 5), 0),
            lambda img: cv2.GaussianBlur(img, (7, 7), 0),
            lambda img: cv2.medianBlur(img, 3),
            lambda img: cv2.medianBlur(img, 5),
            lambda img: cv2.bilateralFilter(img, 9, 75, 75),
        ])
        return blr

class Translation(Transformation):
    def init_id(self):
        self.category = 'geometrical'
        self.name = 'translation'

    def mutate(self, seed):
        x = seed['x']
        img = self.torch2cv(x)
        params = self.extent()
        rows, cols, ch = img.shape
        M = np.float32([[1, 0, params[0]], [0, 1, params[1]]])
        x_ = cv2.warpAffine(img, M, (cols, rows))
        return self.cv2torch(x_), seed['z']

    def extent(self):
        params = [np.random.randint(-3, 4) * 5, np.random.randint(-3, 4) * 5]
        return params

class Scale(Transformation):
    def init_id(self):
        self.category = 'geometrical'
        self.name = 'scale'

    def mutate(self, seed):
        x = seed['x']
        img = self.torch2cv(x)
        ext = self.extent()
        res = cv2.resize(img, None, fx=ext, fy=ext, interpolation=cv2.INTER_CUBIC)
        if ext <= 1:
            x_ = self.cv_pad(img, res)
        else:
            H, W = img.shape[:2]
            top = (res.shape[0] - H) // 2
            left = (res.shape[1] - W) // 2
            x_ = res[top:top+H, left:left+W]
        return self.cv2torch(x_), seed['z']

    def extent(self):
        ext = np.random.choice(list(np.arange(0.5, 1.2, 0.05)))
        return ext

class Rotation(Transformation):
    def init_id(self):
        self.category = 'geometrical'
        self.name = 'rotation'

    def mutate(self, seed):
        x = seed['x']
        img = self.torch2cv(x)
        ext = self.extent()
        rows, cols, ch = img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), ext, 1)
        x_ = cv2.warpAffine(img, M, (cols, rows))
        return self.cv2torch(x_), seed['z']

    def extent(self):
        ext = np.random.choice(list(range(-180, 180)))
        return ext

class Shear(Transformation):
    def init_id(self):
        self.category = 'geometrical'
        self.name = 'shear'

    def mutate(self, seed):
        x = seed['x']
        img = self.torch2cv(x)
        rows, cols, ch = img.shape
        ext = self.extent()
        factor = ext * (-1.0)
        M = np.float32([[1, factor, 0], [0, 1, 0]])
        dst = cv2.warpAffine(img, M, (cols, rows))
        x_ = self.cv_pad(img, dst)
        return self.cv2torch(x_), seed['z']

    def extent(self):
        ext = np.random.choice(list(range(-2, 2)))
        return ext

class Reflection(Transformation):
    def init_id(self):
        self.category = 'geometrical'
        self.name = 'reflection'

    def mutate(self, seed):
        x = seed['x']
        img = self.torch2cv(x)
        x_ = cv2.flip(img, self.extent())
        return self.cv2torch(x_), seed['z']

    def extent(self):
        ext = np.random.randint(-1, 2)
        return ext

class Weather(Transformation):
    def mutate(self, seed):
        x = seed['x']
        img = self.torch2cv(x)
        x_ = self.trans(images=[img])[0]
        return self.cv2torch(x_), seed['z']

class Cloud(Weather):
    def init_id(self):
        self.category = 'style'
        self.name = 'cloud'
        self.trans = iaa.Clouds()

class Fog(Weather):
    def init_id(self):
        self.category = 'style'
        self.name = 'fog'
        self.trans = iaa.Fog()

class Snow(Weather):
    def init_id(self):
        self.category = 'style'
        self.name = 'snow'
        self.trans = iaa.Snowflakes(flake_size=(0.1, 0.4), speed=(0.01, 0.05))

class Rain(Weather):
    def init_id(self):
        self.category = 'style'
        self.name = 'rain'
        self.trans = iaa.Rain(speed=(0.1, 0.3))