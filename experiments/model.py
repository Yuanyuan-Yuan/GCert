#####################################################################
# This script provides certified models in our evaluation.          #
# The implementation is crafted to support being incorporated       #
# into GenProver/ExactLine.                                         #
#####################################################################

import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view([-1] + list(self.shape))

class Negate(nn.Module):
    def __init__(self):
        super(Negate, self).__init__()

    def forward(self, x):
        return -x

class AddOne(nn.Module):
    def __init__(self):
        super(AddOne, self).__init__()

    def forward(self, x):
        return x + 1

class CatTwo(nn.Module):
    def __init__(self):
        super(CatTwo, self).__init__()

    def forward(self, tp):
        (x, y) = tp
        return torch.cat([x, y], dim=1)

class ParSum(nn.Module):
    def __init__(self, net1, net2):
        super(ParSum, self).__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, x, just_left = False):
        r1 = self.net1(x)
        if just_left:
            return r1
        r2 = self.net2(x)
        return r1 + r2

class ConvGeneratorSeq(nn.Module):
    def __init__(self, nc=1, dim=100):
        super(ConvGeneratorSeq, self).__init__()
        nf = 4
        self.dim = dim
        self.net = nn.Sequential(OrderedDict([
            ('seq_0', View([dim, 1, 1])),
            ('seq_1',
            nn.Sequential(
                nn.ConvTranspose2d(dim, nf * 4, 4, 1, 0),
                nn.BatchNorm2d(nf * 4),
                nn.ReLU(),
            )), # upc1
            ('seq_2',
            nn.Sequential(
                nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1),
                nn.BatchNorm2d(nf * 2),
                nn.ReLU()
            )), # upc2
            ('seq_3',
            nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1),
                nn.BatchNorm2d(nf),
                nn.ReLU()
            )), # upc3
            ('seq_4',
            nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.ReLU(),
                Negate(),
                AddOne(),
                nn.ReLU()
            ))
        ]))

    def forward(self, x):
        out = self.net(x)
        return out

class ConvGeneratorSeq32(nn.Module):
    def __init__(self, nc=1, dim=100):
        super(ConvGeneratorSeq32, self).__init__()
        nf = 16
        self.dim = dim
        self.net = nn.Sequential(OrderedDict([
            ('seq_0', View([dim, 1, 1])),
            ('seq_1',
            nn.Sequential(
                nn.ConvTranspose2d(dim, nf * 4, 4, 1, 0),
                nn.BatchNorm2d(nf * 4),
                nn.ReLU(),
            )), # upc1
            ('seq_2',
            nn.Sequential(
                nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1),
                nn.BatchNorm2d(nf * 2),
                nn.ReLU()
            )), # upc2
            ('seq_3',
            nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1),
                nn.BatchNorm2d(nf),
                nn.ReLU()
            )), # upc3
            ('seq_4',
            nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.ReLU(),
                Negate(),
                AddOne(),
                nn.ReLU()
            ))
        ]))

    def forward(self, x):
        out = self.net(x)
        return out

class ConvGeneratorSeq64(nn.Module):
    def __init__(self, nc=1, dim=100):
        super(ConvGeneratorSeq64, self).__init__()
        nf = 16
        self.dim = dim
        self.net = nn.Sequential(OrderedDict([
            ('seq_0', View([dim, 1, 1])),
            ('seq_1',
            nn.Sequential(
                nn.ConvTranspose2d(dim, nf * 4, 4, 1, 0),
                nn.BatchNorm2d(nf * 4),
                nn.ReLU(),
            )), # upc1
            ('seq_2',
            nn.Sequential(
                nn.ConvTranspose2d(nf * 4, nf * 4, 4, 2, 1),
                nn.BatchNorm2d(nf * 4),
                nn.ReLU()
            )), # upc2
            ('seq_3',
            nn.Sequential(
                nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1),
                nn.BatchNorm2d(nf * 2),
                nn.ReLU()
            )), # upc3
            ('seq_4',
            nn.Sequential(
                nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1),
                nn.BatchNorm2d(nf),
                nn.ReLU()
            )), # upc4
            ('seq_5',
            nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.ReLU(),
                Negate(),
                AddOne(),
                nn.ReLU()
            ))
        ]))

    def forward(self, x):
        out = self.net(x)
        return out

class RecogSeq64(nn.Module):
    def __init__(self, nc=3, dim=100):
        super(RecogSeq64, self).__init__()
        nf = 16
        self.dim = dim
        self.net = nn.Sequential(OrderedDict([
            ('seq_0', CatTwo()),
            # 64 x 64
            ('seq_1',
            nn.Sequential(
                nn.Conv2d(nc * 2, nf, 4, 2, 1),
                nn.BatchNorm2d(nf),
                nn.ReLU(),
            )), # upc1
            # 32 x 32
            ('seq_2',
            nn.Sequential(
                nn.Conv2d(nf, nf * 2, 4, 2, 1),
                nn.BatchNorm2d(nf * 2),
                nn.ReLU()
            )), # upc2
            # 16 x 16
            ('seq_3',
            nn.Sequential(
                nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
                nn.BatchNorm2d(nf * 4),
                nn.ReLU()
            )), # upc3
            # 8 x 8
            ('seq_4',
            nn.Sequential(
                nn.Conv2d(nf * 4, nf * 4, 4, 2, 1),
                nn.BatchNorm2d(nf * 4),
                nn.ReLU()
            )), # upc4
            # 4 x 4
            ('seq_5',
            nn.Sequential(
                nn.Conv2d(nf * 4, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.ReLU(), 
            )),
            # 1 x 1
            ('seq_6', View([dim])),
            ('seq_7',
            nn.Sequential(
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )),
        ]))

    def forward(self, tp):
        out = self.net(tp)
        return out

class RecogSeq32(nn.Module):
    def __init__(self, nc=3, dim=100):
        super(RecogSeq32, self).__init__()
        nf = 16
        self.dim = dim
        self.net = nn.Sequential(OrderedDict([
            ('seq_0', CatTwo()),
            # 32 x 32
            ('seq_1',
            nn.Sequential(
                nn.Conv2d(nc * 2, nf, 4, 2, 1),
                nn.BatchNorm2d(nf),
                nn.ReLU(),
            )), # upc1
            # 16 x 16
            ('seq_2',
            nn.Sequential(
                nn.Conv2d(nf, nf * 2, 4, 2, 1),
                nn.BatchNorm2d(nf * 2),
                nn.ReLU()
            )), # upc2
            # 8 x 8
            ('seq_3',
            nn.Sequential(
                nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
                nn.BatchNorm2d(nf * 4),
                nn.ReLU()
            )), # upc3
            # 4 x 4
            ('seq_4',
            nn.Sequential(
                nn.Conv2d(nf * 4, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.ReLU(), 
            )),
            # 1 x 1
            ('seq_5', View([dim])),
            ('seq_6',
            nn.Sequential(
                nn.Linear(dim, 1),
                nn.Sigmoid()
            )),
        ]))

    def forward(self, tp):
        out = self.net(tp)
        return out

class F1(nn.Module):
    def __init__(self, n_class=10, dim=10):
        super(F1, self).__init__()
        self.dim = dim
        self.net = nn.Sequential(OrderedDict([
            ('seq_0', View([32 * 32])),
            ('seq_1',
            nn.Sequential(
                nn.Linear(32 * 32, dim),
                # nn.BatchNorm1d(dim),
                nn.ReLU(),
            )),
            ('seq_2',
            nn.Sequential(
                nn.Linear(dim, dim),
                # nn.BatchNorm1d(dim),
                nn.ReLU(),
            )),
            ('seq_3',
            nn.Sequential(
                nn.Linear(dim, n_class),
            )),
        ]))

    def forward(self, x):
        out = self.net(x)
        return out

class F2(nn.Module):
    def __init__(self, nc=1, n_class=10, dim=10):
        super(F2, self).__init__()
        self.dim = dim
        nf = 16
        self.net = nn.Sequential(OrderedDict([
            # 32 x 32
            ('seq_0',
            nn.Sequential(
                nn.Conv2d(nc, nf, 4, 2, 1),
                nn.BatchNorm2d(nf),
                nn.ReLU(),
            )),
            # 16 x 16
            ('seq_1',
            nn.Sequential(
                nn.Conv2d(nf, nf * 2, 4, 4, 0),
                nn.BatchNorm2d(nf * 2),
                nn.ReLU(),
            )),
            # 4 x 4
            ('seq_2', View([nf * 2 * 4 * 4])),
            ('seq_3',
            nn.Sequential(
                nn.Linear(nf * 2 * 4 * 4, n_class),
            )),
        ]))

    def forward(self, x):
        out = self.net(x)
        return out

class F3(nn.Module):
    def __init__(self, nc=1, n_class=10, dim=10):
        super(F3, self).__init__()
        self.dim = dim
        nf = 16
        self.net = nn.Sequential(OrderedDict([
            # 32 x 32
            ('seq_0',
            nn.Sequential(
                nn.Conv2d(nc, nf, 4, 2, 1),
                nn.BatchNorm2d(nf),
                nn.ReLU(),
            )),
            # 16 x 16
            ('seq_1',
            nn.Sequential(
                nn.Conv2d(nf, nf * 2, 4, 2, 1),
                nn.BatchNorm2d(nf * 2),
                nn.ReLU(),
            )),
            # 8 x 8
            ('seq_2',
            nn.Sequential(
                nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
                nn.BatchNorm2d(nf * 4),
                nn.ReLU(),
            )),
            # 4 x 4
            ('seq_3',
            nn.Sequential(
                nn.Conv2d(nf * 4, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.ReLU(),
            )),
            # 1 x 1
            ('seq_4', View([dim])),
            ('seq_5',
            nn.Sequential(
                nn.Linear(dim, dim),
                # nn.BatchNorm1d(dim),
                nn.ReLU(),
            )),
            ('seq_6',
            nn.Sequential(
                nn.Linear(dim, n_class),
            )),
        ]))

    def forward(self, x):
        out = self.net(x)
        return out