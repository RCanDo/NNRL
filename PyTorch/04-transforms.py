#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Transforms
subtitle:
version: 1.0
type: module
keywords: [PyTorch, NN, AI]   # there are always some keywords!
description: |
remarks:
    - my version of v1
todo:
sources:
    - title: Transforms
      link: https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html#transforms
file:
    usage:
        interactive: True
        terminal: True
    name: 04-transforms.py
    path: ~/Projects/AIML/NNRL/PyTorch
    date: 2022-03-16
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""


#%% Transforms
"""
All TorchVision datasets have two parameters
that accept callables containing the transformation logic:
 transform to modify the features
 target_transform to modify the labels
The
 torchvision.transforms
module offers several commonly-used transforms out of the box.

The FashionMNIST features are in PIL Image format, and the labels are integers.
For training, we need the features as normalized tensors,
and the labels as one-hot encoded tensors.
To make these transformations, we use  ToTensor  and  Lambda.
"""
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)

#%% ToTensor()
"""
ToTensor converts a  PIL image  or  NumPy ndarray  into a  FloatTensor.
and scales the image’s pixel intensity values in the range [0., 1.]
"""
#%% Lambda Transforms
"""
Lambda transforms apply any user-defined lambda function.
Here, we define a function to turn the integer into a one-hot encoded tensor.
It first creates a zero tensor of size 10 (the number of labels in our dataset)
and calls  scatter_  which assigns a value=1 on the index as given by the label y.
"""
target_transform = Lambda(
    lambda y: torch.zeros(10, dtype=torch.float) \
        .scatter_(dim=0, index=torch.tensor(y), value=1)
    )

#%%
target_transform(3)
target_transform(torch.tensor(3))

#%%
#%% help(tensor.scatter_)
"""
Writes all values from the tensor src into self at the indices specified in the index tensor.
For each value in src, its output index is specified by its index in src for dimension != dim
and by the corresponding value in index for dimension = dim.   oooops...

dim (int): the axis along which to index
index (LongTensor): the indices of elements to scatter,
  can be either empty or the same size of src.
  When empty, the operation returns identity
src (Tensor): the source element(s) to scatter, in case `value` is not specified
value (float): the source element(s) to scatter, in case `src` is not specified (pytorch 1.4.0, not in online help)
reduce (str, optional): reduction operation to apply, can be either 'add' or 'multiply' (pytorch 1.11.0, not in inline help).
"""
src = torch.arange(1, 11).reshape((2, 5))
src
index = torch.tensor([[0, 1, 2, 0]])
index

torch.zeros(3, 5, dtype=src.dtype).scatter_(0, index, src)

index = torch.tensor([[0, 1, 2], [0, 1, 4]])
torch.zeros(3, 5, dtype=src.dtype).scatter_(1, index, src)

torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
           1.23, reduce='multiply')

torch.full((2, 4), 2.).scatter_(1, torch.tensor([[2], [3]]),
           1.23, reduce='add')

#%%
