# Implementation

This folder provides implementations and examples of regulating generative models with independence and continuity.

## Continuity

To enforce the continuity, you need to add an extra training objective. Below, we show how to train a conventional GAN with regulation of continuity.

```python
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
```

## Independence

The independence is ensured from the following two aspects.

### Global Mutations

For global mutations, different mutations are represented as *orthogonal* directions in the latent space. This is achieved using SVD; see details in `independence.py`.

Below is an example of getting global mutating directions

```python
J = Jacobian(G, z)
# `G` is the generative model and `z` is the latent point
directions = get_direction(J, None)
```

### Local Mutations

For local mutations, besides representing different mutations as orthogonal directions, we also ensure that only the selected local region is mutated. This is achieved by projecting mutating directions of the local region into non-mutating directions of the background.

Before performing local mutation, you need to manualy set the foreground and backgroud indexes. Below is an example of mutating eyes for ffhq images.

```python
COORDINATE_ffhq = {
    'left_eye': [120, 95, 20, 38],
    'right_eye': [120, 159, 20, 38],
    'eyes': [120, 128, 20, 115],
    'nose': [142, 131, 40, 46],
    'mouth': [184, 127, 30, 70],
    'chin': [217, 130, 42, 110],
    'eyebrow': [126, 105, 15, 118],
}

def get_mask_by_coordinates(image_size, coordinate):
    """Get mask using the provided coordinates."""
    mask = np.zeros([image_size, image_size], dtype=np.float32)
    center_x, center_y = coordinate[0], coordinate[1]
    crop_x, crop_y = coordinate[2], coordinate[3]
    xx = center_x - crop_x // 2
    yy = center_y - crop_y // 2
    mask[xx:xx + crop_x, yy:yy + crop_y] = 1.
    return mask

coords = COORDINATE_ffhq['eyes']
mask = get_mask_by_coordinates(256, coordinate=coords)
foreground_ind = np.where(mask == 1)
background_ind = np.where((1 - mask) == 1)
directions = get_direction(J, None, foreground_ind, background_ind)
```

## Performing Mutations

Once you get the mutating directions, you can perform mutations in the following way.

```python
delta = 1.0
for i in range(len(directions)):
    v = directions[i]
    x_ = G(z + delta * v)
```

`delta` controls the extent of the mutation. `x_` is the mutated input using the `i`-th mutating direction.