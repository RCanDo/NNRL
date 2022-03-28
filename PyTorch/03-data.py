#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Datasets & DataLoaders
subtitle:
version: 1.0
type: module
keywords: [PyTorch, NN, AI]   # there are always some keywords!
description: |
remarks:
    - my version of v1
todo:
sources:
    - title: Datasets & DataLoaders
      link: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
file:
    usage:
        interactive: True
        terminal: True
    name: 03-data.py
    path: ~/Projects/AIML/NNRL/PyTorch
    date: 2022-03-13
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""

#%%
"""
PyTorch provides two data primitives:

 torch.utils.data.Dataset -- stores the samples and their corresponding labels
 torch.utils.data.DataLoader -- wraps an iterable around the Dataset to enable easy access to the samples.

PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST)
that subclass torch.utils.data.Dataset and implement functions specific to the particular data.
They can be used to prototype and benchmark your model.
You can find them here:
    Image Datasets, Text Datasets, and Audio Datasets.
"""

#%% Loading a Dataset
"""
Here is an example of how to load the Fashion-MNIST dataset from TorchVision.
Fashion-MNIST is a dataset of Zalando’s article images consisting of
60,000 training examples and 10,000 test examples.
Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

We load the FashionMNIST Dataset with the following parameters:

`root` is the path where the train/test data is stored,
`train` specifies training or test dataset,
`download=True` downloads the data from the internet if it’s not available at `root`.
`transform` and `target_transform` specify the feature and label transformations
"""

#%%
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)
#%%
training_data
type(training_data)     # torchvision.datasets.mnist.FashionMNIST
len(training_data)
dir(training_data)      # ... see 01-quickstart.py
training_data.targets   # tensor([9, 0, 0,  ..., 3, 0, 5])
training_data.classes   #

training_data[1]    # tuple, length = 2

#%% Iterating and Visualizing the Dataset
"""
We can index Datasets manually like a list:
    training_data[index].
We use matplotlib to visualize some samples in our training data.
"""
labels_map = {i: c for i, c in enumerate(training_data.classes)}
    # 0: "T-Shirt",
    # 1: "Trouser",
    # 2: "Pullover",
    # 3: "Dress",
    # 4: "Coat",
    # 5: "Sandal",
    # 6: "Shirt",
    # 7: "Sneaker",
    # 8: "Bag",
    # 9: "Ankle Boot",

figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()  # .item() = take nuber out of tensor

    img, label = training_data[sample_idx]  #!!!

    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()

img.shape   # torch.Size([1, 28, 28])
label       # 3

#%% Creating a Custom Dataset for your files
"""
A custom Dataset class must implement three functions:
__init__,  __len__  and  __getitem__.
Take a look at this implementation;
the FashionMNIST images are stored in a directory  `img_dir`,
and their labels are stored separately in a CSV file  `annotations_file`:

tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9

In the next sections, we’ll break down what’s happening in each of these functions.
"""
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


#%% Preparing your data for training with DataLoaders
"""
The Dataset retrieves our dataset’s features and labels one sample at a time.
While training a model, we typically want to pass samples in “minibatches”,
reshuffle the data at every epoch to reduce model overfitting,
and use Python’s multiprocessing to speed up data retrieval.

DataLoader is an iterable that abstracts this complexity for us in an easy API.
"""
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

#%% Iterate through the DataLoader
"""
We have loaded that dataset into the DataLoader
and can iterate through the dataset as needed.
Each iteration below returns a batch of train_features and train_labels
(containing batch_size=64 features and labels respectively).
Because we specified shuffle=True,
after we iterate over all batches the data is shuffled
(for finer-grained control over the data loading order, take a look at Samplers).
"""
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
train_features, train_labels = next(train_dataloader)   #! TypeError: 'DataLoader' object is not an iterator

from itertools import islice
train_features, train_labels = next(islice(train_dataloader, 1))

# it's a BATCH of data (64 items)
train_features.size()   # torch.Size([64, 1, 28, 28])
train_labels.size()     # torch.Size([64])

# 1st image and label
train_features[0]
train_features[0].shape # torch.Size([1, 28, 28])
img = train_features[0].squeeze()    # get rid of unnecessary dimensions
img.shape               # torch.Size([28, 28])
label = train_labels[0] # tensor(4)
label.item()            # 4

plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label} -- {labels_map[label.item()]}")




#%%
