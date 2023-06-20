# GCert
Research Artifact of USENIX Security 2023 Paper: *Precise and Generalized Robustness Certification for Neural Networks*

Preprint: https://arxiv.org/pdf/2306.06747.pdf


## Installation

- Build from source code

    ```setup
    git clone https://github.com/Yuanyuan-Yuan/GCert
    cd GCert
    pip install -r requirements.txt
    ```

## Structure

This repo is organized as follows:

- `implementation` - This folder provides implementations and examples of regulating
generative models with continuity and independence. See detailed documents [here](https://github.com/Yuanyuan-Yuan/GCert/tree/main/implementation)

- `experiments` - This folder provides scripts of our evaluations. See detailed documents [here](https://github.com/Yuanyuan-Yuan/GCert/tree/main/experiments)

- `frameworks` - GCert is incorporated into three conventional certification frameworks (i.e.,
AI2/Eran, GenProver, and ExactLine). This folder provides the scripts for configurations; see
detailed documents [here](https://github.com/Yuanyuan-Yuan/GCert/tree/main/frameworks)

- `data` - This folder provides scripts for data processing and shows examples of some data samples. See detailed documents [here](https://github.com/Yuanyuan-Yuan/GCert/tree/main/data).

## Data

The following four datasets are considered in our evaluation.

- MNIST - We use the dataset provided by Pytorch (see [here](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)). Note that the original image size is $1 \times 27 \times 27$. We resize the image size to $1 \times 32 \times 32$.

- CIFAR10 - We use the dataset provided by Pytorch (see [here](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)).

- Driving - The images can be downloaded [here](https://github.com/SullyChen/driving-datasets). We provide several examples in `data/driving` folder. The dataset class can be implemented using the [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) class in Pytorch.

- CelebA - The official dataset can be downloaded [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Once downloading the dataset, you can use `data/celeba_crop128/celeba_process.py` to process the dataset, which splits the dataset into different subfolders and crop and resize the faces into $128 \times 128$. Several processed examples are given in the `data/celeba_crop128/train` folder. We also provide the mapping between file names and human IDs in `data/CelebA_ID_to_name.json` and `data/CelebA_name_to_ID.json`.  
The dataset class is implemented in `experiments/dataset.py`.

## Frameworks

GCert is incorporated into the following three frameworks for certification.

### GenProver 

The official implementation of GenProver is provided [here](https://openreview.net/forum?id=HJxRMlrtPH).

After downloading the code, you need to modify the following scripts in the projects:

- `frameworks/GenProver/components.py` - GenProver is implemented based on [DiffAI](https://github.com/eth-sri/diffai) and `components.py` re-implements different Pytorch `nn` modules with `InferModule` of DiffAI. We modified the implementations of several modules (mostly the `BatchNorm` module) to better fit the implementations in Pytorch. You can replace the original `components.py` with our provided one.

- `frameworks/GenProver/genmodels.py` - We added implementations (with DiffAI modules) of our models in this script. You can replace the original `genmodels.py` with our provided one.

Note that in order to load models trained with Pytorch, you need to do the following:

1. Implement the model following the examples given in `experiments/model.py`. We suggest implementing the model with `nn.Sequential()` and hard-coding the name for each `nn.Sequential()`.

2. Implement every operation as a class inherited from Pytorch `nn` module. For example, the `torch.cat()` operation should be implement as `class CatTwo(nn.Module)` in `experiments/model.py`; see examples in `experiments/model.py`.

3. Implement the corresponding class following DiffAI in `frameworks/GenProver/components.py`. For example, for the `class CatTwo(nn.Module)` in `experiments/model.py`, you should implement a `class CatTwo(InferModule)` in `components.py`; more examples are given in `components.py`.

4. When loading the trained weights, you need to convert the key in `state_dict`. We provide the implementation and examples in `frameworks/GenProver/load_model.py`.

### ExactLine

We use the ExactLine implemented by authors of GenProver. The source code can be downloaded [here](https://openreview.net/forum?id=HJxRMlrtPH).

The implmentation of ExactLine and GenProver are almost the same, except that GenProver merges segments in intermediate outputs as box/polyhedra. Thus, to use ExactLine, you only need to set

```python
use_clustr = None
```

in the implementation of GenProver.

### AI2/ERAN

The official implementation is provided [here](https://github.com/eth-sri/eran). In our experiments, we use the adaptor provided by [VeriGauge](https://github.com/AI-secure/VeriGauge) to set up AI2/ERAN.

[VeriGauge](https://github.com/AI-secure/VeriGauge) and [AI2/ERAN](https://github.com/eth-sri/eran) are well implemented and documented; you can smoothly set up everything following their instructions.

## Implementation

This folder provides implementations and examples of regulating generative models with independence and continuity.

### Continuity

To enforce the continuity, you need to add an extra training objective. See more details in `implementation/continuity.py`. Below, we show how to train a conventional GAN with regulation of continuity.

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

### Independence

The independence is ensured from the following two aspects.

#### Global Mutations

For global mutations, different mutations are represented as *orthogonal* directions in the latent space. This is achieved using SVD; see details in `implementation/independence.py`.

Below is an example of getting global mutating directions

```python
J = Jacobian(G, z)
# `G` is the generative model and `z` is the latent point
directions = get_direction(J, None)
```

#### Local Mutations

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

The coordinates are provided by authors of [LowRankGAN](https://github.com/zhujiapeng/resefa/blob/main/coordinate.py).

### Performing Mutations

Once you get the mutating directions, you can perform mutations in the following way.

```python
delta = 1.0
for i in range(len(directions)):
    v = directions[i]
    x_ = G(z + delta * v)
```

`delta` controls the extent of the mutation. `x_` is the mutated input using the `i`-th mutating direction.

## Experiments

This folder provides scripts for our evaluations.

### Models and Datasets

- `experiments/dataset.py` - We implement two Pytorch Dataset classes for CelebA. `CelebARecog` is used for training face recognition models. `CelebAGen` is employed for training face image generator.

- `experiments/model.py` - We implement our models in this script. In accordance to requirements of GenProver/ExactLine, the implementations are carefully crafted. See details in [GenProver](https://github.com/Yuanyuan-Yuan/GCert/tree/main/frameworks/GenProver).

- `experiments/face_recognition.py` - Our face recognition model takes a tuple of two images as one input and predicts whether the two faces are from the same person. This script implements how we train the face recognition model.

See [data](https://github.com/Yuanyuan-Yuan/GCert/tree/main/data) for how to download and process the datasets.

### Mutations

#### Geometrical

- `experiments/augment_geometrical.py` - This script shows how we augment the training data with different geometrical (affine) mutations. In brief, this is achieved by applying the mutation in runtime.

Pytorch `transforms` module supports randomly applying affine mutations on each input, see implementation below.

```python
transforms.RandomAffine(
                degrees=30,
                # translate=(0.3, 0.3),
                # scale=(0.75, 1.2),
                # shear=(0.2)
            ),
```

We also provide implementations of different mutations in `experiments/mutation.py`. Below is the example of rotation.

```python
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
        # Set the maximal extent of mutations here
        return ext
```

You can also augment the training data with rotation (such that rotation can be decomposed from the latent space of the generative model) in the follow way.

```python
from mutation import Rotation

for epoch in range(num_epoch):
    for (image, *_) in dataloader:
        image_ = Rotation.mutate(image)
        # Then use `image_` to train the generative model
```

#### Perceptual-Level

For perceptual-level mutations, since they are extracted from the perception variations from natural images, you don not need to do anything; just train a standard generative model. See `implementation/independence.py` for how to obtain perceptual-level mutations.

#### Stylized

For stylized mutations, you need to train the generative model following the cycle-consistency (which is proposed in CycleGAN). The official Pytorch implementation of CycleGAN is provided [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). You can smoothly set up everything following the official documents.

For different artistical styles, we use the style files provided [here](https://github.com/rgeirhos/Stylized-ImageNet).

For weather-filters, we use the simulated filters provided by [imgaug](https://github.com/aleju/imgaug). The implementations are given in `experiments/mutation.py`. Below is an example of the foggy mutation.

```python
import imgaug.augmenters as iaa

class Weather(Transformation):
    def mutate(self, seed):
        x = seed['x']
        img = self.torch2cv(x)
        x_ = self.trans(images=[img])[0]
        return self.cv2torch(x_), seed['z']

class Fog(Weather):
    def init_id(self):
        self.category = 'style'
        self.name = 'fog'
        self.trans = iaa.Fog()
```

### Evalution Tools

- `experiments/rectangle.py` - This script implements how to calculate the minimal enclosing rectangle for assessing the geometrical properties.

- `experiments/synthetic_data.py` - This script implements the synthetic dataset of our ablation study. You can directly use the `SyntheticDataset` class as one Pytorch dataset class.

## Acknowledgement

We sincerely thank authors of the following projects for open-sourcing their code, which greatly help us develop GCert.

- GenProver/DiffAI: https://github.com/eth-sri/diffai

- AI2/ERAN: https://github.com/eth-sri/eran

- VeriGauge: https://github.com/AI-secure/VeriGauge

- ExactLine: https://github.com/95616ARG/SyReNN

- LowRankGAN: https://github.com/zhujiapeng/LowRankGAN

## Citation

If GCert is helpful for your research, please consider cite our work as follows:

```bib
@inproceedings{yuan2023precise,
  title={Precise and Generalized Robustness Certification for Neural Networks},
  author={Yuan, Yuanyuan and Wang, Shuai and Su, Zhendong},
  booktitle={32nd USENIX Security Symposium (USENIX Security 23)},
  year={2023}
}
```

If you have any questions, feel free to contact Yuanyuan (yyuanaq@cse.ust.hk).