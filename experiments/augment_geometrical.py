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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


from model import *

os.makedirs('images', exist_ok=True)
os.makedirs('ckpt', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
parser.add_argument('--cls', type=int, default=0, help='selected class')
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
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

# Initialize generator and discriminator
generator = ConvGeneratorSeq()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()

# Configure data loader
os.makedirs('./data/mnist', exist_ok=True)

dataset_full = datasets.MNIST(
        './data/mnist',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomAffine(
                degrees=30,
                # translate=(0.3, 0.3),
                # scale=(0.75, 1.2),
                # shear=(0.2)
            ),

            transforms.Resize(args.img_size),
            transforms.ToTensor(), 
            ]
        ),
    )
# Selecting classes 7, 2, 5 and 6
if args.cls != -1:
    idx = (dataset_full.targets == args.cls)
    dataset_full.targets = dataset_full.targets[idx]
    dataset_full.data = dataset_full.data[idx]

dataloader = torch.utils.data.DataLoader(
    dataset_full,
    batch_size=args.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


########################################
# This is original training objective. #
########################################
def compute_gradient_penalty(D, real_samples, fake_samples):
    '''Calculates the gradient penalty loss for WGAN GP'''
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

########################################
# Regulation with continuity.          #
########################################
def continuity(generator):
    # if the latent space follows uniform distribution
    z1 = Tensor(np.random.uniform(-1, 1, (batch_size, latent_dimension)))
    z2 = Tensor(np.random.uniform(-1, 1, (batch_size, latent_dimension)))
    # # if the latent space follows normal distribution
    # z1 = Tensor(np.random.normal(0, 1, (batch_size, latent_dimension)))
    # z2 = Tensor(np.random.normal(0, 1, (batch_size, latent_dimension)))
    G1 = generator(z1)
    G2 = generator(z2)
    gamma = random.uniform(0, 1)
    z = torch.lerp(z1, z2, gamma)
    # an `intermediate point` between z1 and z2
    G = generator(z)
    penality = (gamma * G2 - G - (1 - gamma) * G1).square().mean()
    return penality

########################################
# Training                             #
########################################
for epoch in range(args.n_epochs):
    d_loss_list = []
    g_loss_list = []

    generator.train()
    discriminator.train()
    for i, (imgs, *_) in enumerate(tqdm(dataloader)):

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()
        # Sample noise as generator input
        z = Variable(Tensor(np.random.uniform(-1, 1, (imgs.shape[0], args.latent_dim))))
        # Generate a batch of images
        fake_imgs = generator(z)

        # Real images
        real_validity = discriminator(real_imgs)
        # Fake images
        fake_validity = discriminator(fake_imgs)
        # Gradient penalty
        gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
        # Adversarial loss
        d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + args.lambda_gp * gradient_penalty

        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()

        # Train the generator every n_critic steps
        if i % args.n_critic == 0:

            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            fake_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            # Train on fake images
            fake_validity = discriminator(fake_imgs)
            g_loss = -torch.mean(fake_validity)

            continuity_penalty = continuity(generator)
            
            (g_loss + continuity_penalty).backward(retain_graph=True)
            optimizer_G.step()

            d_loss_list.append(d_loss.item())
            g_loss_list.append(g_loss.item())

    print(
        '[Epoch %d/%d] [D loss: %f] [G loss: %f]'
        % (epoch, args.n_epochs, np.mean(d_loss_list), np.mean(g_loss_list))
    )
    if epoch % args.save_every == 0:
        save_image(fake_imgs.data[:100], 'images/%s/%d_fake.png' % (args.exp_name, epoch), nrow=10, normalize=True)
        save_image(real_imgs.data[:100], 'images/%s/%d_real.png' % (args.exp_name, epoch), nrow=10, normalize=True)
        generator.eval()
        discriminator.eval()
        state = {
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict(),
            'optimizer_G': optimizer_G.state_dict(),
            'optimizer_D': optimizer_D.state_dict()
        }
        torch.save(state, 'ckpt/%s/%d.ckpt' % (args.exp_name, epoch))

save_image(fake_imgs.data[:100], 'images/%s/final_fake.png' % (args.exp_name), nrow=10, normalize=True)
save_image(real_imgs.data[:100], 'images/%s/final_real.png' % (args.exp_name), nrow=10, normalize=True)
generator.eval()
discriminator.eval()
state = {
    'generator': generator.state_dict(),
    'discriminator': discriminator.state_dict(),
    'optimizer_G': optimizer_G.state_dict(),
    'optimizer_D': optimizer_D.state_dict()
}
torch.save(state, 'ckpt/%s/final.ckpt' % (args.exp_name))