############################################################
# In this script we show how our continuity regulation     #
# is applied on the training stage of generative models.   #
############################################################

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# This script assumes you are using Pytorch

cuda = False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

batch_size = 32
latent_dimension = 50

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


##########################################################
# Below we show how the function `continuity` is used in #
# standard training process of GAN.                      #
##########################################################

# you can use any generator
generator = None
discriminator = None
dataloader = None
# replace the above with your custom ones


learning_rate = 0.01
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)

bce = torch.nn.BCELoss()
real = Tensor(np.ones(batch_size, 1))
fake = Tensor(np.zeros(batch_size, 1))

n_epochs = 100
for epoch in range(n_epochs):
    generator.train()
    discriminator.train()
    for i, (images, *_) in enumerate(tqdm(dataloader)):

        images = Tensor(images)

        optimizer_G.zero_grad()
        z = Tensor(np.random.uniform(-1, 1, (batch_size, latent_dimension)))
        # or: Tensor(np.random.normal(0, 1, (batch_size, latent_dimension)))
        G = generator(z)
        g_loss = bce(discriminator(G), real)

        g_loss += continuity(generator)
        # just add this one line :D

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = bce(discriminator(images), real)
        fake_loss = bce(discriminator(G), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()