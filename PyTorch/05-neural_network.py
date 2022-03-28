#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Build the Neural Network
subtitle:
version: 1.0
type: module
keywords: [PyTorch, NN, AI]   # there are always some keywords!
description: |
remarks:
    - my version of v1
todo:
sources:
    - title: Build the Neural Network
      link: https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
file:
    usage:
        interactive: True
        terminal: True
    name: 05-neural_networks.py
    path: ~/Projects/AIML/NNRL/PyTorch
    date: 2022-03-16
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""

#%% Build the Neural Network
"""
Every module in PyTorch subclasses the  nn.Module.
A neural network is a module itself that consists of other modules (layers).
This nested structure allows for building and managing complex architectures easily.

In the following sections, we’ll build a neural network to classify images in the FashionMNIST dataset.
"""
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#%% Get Device for Training
"""
We want to be able to train our model on a hardware accelerator like the GPU,
if it is available.
Let’s check to see if torch.cuda is available, else we continue to use the CPU.
"""
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

#%% Define the Class
"""
We define our neural network by subclassing nn.Module,
and initialize the neural network layers in __init__.
Every  nn.Module  subclass implements the operations on input data in the forward method.
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features=28*28, out_features=512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            )

    def forward(self, x):
        """!!!  Do not call model.forward() directly  !!!!
        """
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#%% some prediction from the model (untrained yet!)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)   # direct prediction

pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")

#%%
model

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")



#%%




#%%
