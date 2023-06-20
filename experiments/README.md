# Experiments

This folder provides scripts for our evaluations.

## Models and Datasets

- `dataset.py` - We implement two Pytorch Dataset classes for CelebA. `CelebARecog` is used for training face recognition models. `CelebAGen` is employed for training face image generator.

- `model.py` - We implement our models in this script. In accordance to requirements of GenProver/ExactLine, the implementations are carefully crafted. See details in [GenProver]().

- `face_recognition.py` - Our face recognition model takes a tuple of two images as one input and predicts whether the two faces are from the same person. This script implements how we train the face recognition model.

See [data]() for how to download and process the datasets.

## Mutations

### Geometrical

- `augment_geometrical.py` - This script shows how we augment the training data with different geometrical (affine) mutations. In brief, this is achieved by applying the mutation in runtime.
Pytorch `transforms` module supports randomly applying affine mutations on each input, see implementation below.

```python
transforms.RandomAffine(
                degrees=30,
                # translate=(0.3, 0.3),
                # scale=(0.75, 1.2),
                # shear=(0.2)
            ),
```

We also provide implementations of different mutations in `mutation.py`. Below is the example of rotation.

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

### Perceptual-Level

For perceptual-level mutations, since they are extracted from the perception variations from natural images, you don not need to anything; just train a standard generative model. See `implementation/independence.py` for how to obtain perceptual-level mutations.

### Stylized

For stylized mutations, you need to train the generative model following the cycle-consistency (which is proposed in CycleGAN). The official Pytorch implementation of CycleGAN is provided [here](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). You can smoothly set up everything following the official documents.

For different artistical styles, we the style files provided [here](https://github.com/rgeirhos/Stylized-ImageNet).

For weather-filters, we use the simulated filters provided by [imgaug](https://github.com/aleju/imgaug). The implementations are given in `mutation.py`. Below is an example of the foggy mutation.

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

## Evalution Tools

- `rectangle.py` - This script implements how to calculate the minimal enclosing rectangle for assessing the geometrical properties.

- `synthetic_data.py` - This script implements the synthetic dataset of our ablation study. You can directly use the `SyntheticDataset` class as one Pytorch dataset class.