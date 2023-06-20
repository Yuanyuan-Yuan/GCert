# Data

The following four datasets are considered in our evaluation.

- MNIST - We use the dataset provided by Pytorch (see [here](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)). Note that the original image size is $1 \times 27 \times 27$. We resize the image size to $1 \times 32 \times 32$.

- CIFAR10 - We use the dataset provided by Pytorch (see [here](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html)).

- Driving - The images can be downloaded [here](https://github.com/SullyChen/driving-datasets). We provide several examples in `driving` folder. The dataset class can be implemented using the [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) class in Pytorch.

- CelebA - The official dataset can be downloaded [here](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Once downloading the dataset, you can use `celeba_crop128/celeba_process.py` to process the dataset, which splits the dataset into different subfolders and crop and resize the faces into $128 \times 128$. Several processed examples are given in the `celeba_crop128/train` folder. We also provide the mapping between file names and human IDs in `CelebA_ID_to_name.json` and `CelebA_name_to_ID.json`.  
The dataset class is implemented in `experiments/dataset.py`.