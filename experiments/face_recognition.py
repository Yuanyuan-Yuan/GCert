import argparse
import os
import random
from tqdm import tqdm
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

from model import *
from dataset import CelebARecog

os.makedirs('images', exist_ok=True)
os.makedirs('ckpt', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='face_recog_32', help='experiment name')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--num_tuple', type=int, default=10000)
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--n_critic', type=int, default=3, help='number of training steps for discriminator per iter')
parser.add_argument('--lambda_gp', type=float, default=10, help='loss weight for gradient penalty')
parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')
# parser.add_argument('--sample_interval', type=int, default=500, help='interval betwen image samples')
parser.add_argument('--save_every', type=int, default=50, help='interval betwen image samples')
args = parser.parse_args()
print(args)

os.makedirs('images/%s' % args.exp_name, exist_ok=True)
os.makedirs('ckpt/%s' % args.exp_name, exist_ok=True)

img_shape = (args.channels, args.img_size, args.img_size)

cuda = True if torch.cuda.is_available() else False

# classifier = RecogSeq64()
classifier = RecogSeq32()

if cuda:
    classifier.cuda()

# Configure data loader

train_set = CelebARecog(split='train', num_tuple=args.num_tuple, img_size=args.img_size)
test_set = CelebARecog(split='test', num_tuple=args.num_tuple, img_size=args.img_size)

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=True,
)

# Optimizers
optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr, betas=(args.b1, args.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# ----------
#  Training
# ----------

def accuracy(pred, target):
    is_same = ((pred > 0.5) == target)
    return (is_same.sum() / len(is_same)).item()

mse = nn.MSELoss().cuda()
bce_log = nn.BCEWithLogitsLoss().cuda()
bce = nn.BCELoss().cuda()

def train():
    loss_list = []
    pos_acc_list = []
    neg_acc_list = []
    classifier.train()
    for i, (img, same, diff) in enumerate(tqdm(train_loader)):

        # Configure input
        img = Variable(img.type(Tensor))
        same = Variable(same.type(Tensor))
        diff = Variable(diff.type(Tensor))

        ones = Variable(torch.ones([img.size(0), 1]).type(Tensor))
        zeros = Variable(torch.zeros([img.size(0), 1]).type(Tensor))

        optimizer.zero_grad()

        pred_same = classifier((img, same))
        pred_diff = classifier((img, diff))

        loss_same = bce(pred_same, ones)
        loss_diff = bce(pred_diff, zeros)
        loss = loss_same + loss_diff

        loss.backward()
        optimizer.step()

        pos_acc = accuracy(pred_same, ones)
        neg_acc = accuracy(pred_diff, zeros)
        loss_list.append(loss.item())
        pos_acc_list.append(pos_acc)
        neg_acc_list.append(neg_acc)
    return loss_list, pos_acc_list, neg_acc_list

def test():
    with torch.no_grad():
        loss_list = []
        pos_acc_list = []
        neg_acc_list = []
        classifier.eval()
        for i, (img, same, diff) in enumerate(tqdm(test_loader)):

            # Configure input
            img = Variable(img.type(Tensor))
            same = Variable(same.type(Tensor))
            diff = Variable(diff.type(Tensor))

            ones = Variable(torch.ones([img.size(0), 1]).type(Tensor))
            zeros = Variable(torch.zeros([img.size(0), 1]).type(Tensor))

            pred_same = classifier((img, same))
            pred_diff = classifier((img, diff))

            loss_same = bce(pred_same, ones)
            loss_diff = bce(pred_diff, zeros)
            loss = loss_same + loss_diff

            pos_acc = accuracy(pred_same, ones)
            neg_acc = accuracy(pred_diff, zeros)
            loss_list.append(loss.item())
            pos_acc_list.append(pos_acc)
            neg_acc_list.append(neg_acc)
        return loss_list, pos_acc_list, neg_acc_list


for epoch in range(args.n_epochs):
    loss_list, pos_acc_list, neg_acc_list = train()
    print(
        '[Epoch %d/%d] [loss: %f] [pos acc: %f] [neg acc: %f]'
        % (epoch, args.n_epochs, np.mean(loss_list), np.mean(pos_acc_list), np.mean(neg_acc_list))
    )
    if epoch % args.save_every == 0:
        loss_list, pos_acc_list, neg_acc_list = test()
        print(
            '[Test] [loss: %f] [pos acc: %f] [neg acc: %f]'
            % (np.mean(loss_list), np.mean(pos_acc_list), np.mean(neg_acc_list))
        )
        state = {
            'classifier': classifier.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, 'ckpt/%s/%d.ckpt' % (args.exp_name, epoch))

loss_list, pos_acc_list, neg_acc_list = test()
print(
    '[Test] [loss: %f] [pos acc: %f] [neg acc: %f]'
    % (np.mean(loss_list), np.mean(pos_acc_list), np.mean(neg_acc_list))
)
state = {
    'classifier': classifier.state_dict(),
    'optimizer': optimizer.state_dict(),
}
torch.save(state, 'ckpt/%s/final.ckpt' % (args.exp_name))