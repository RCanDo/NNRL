#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: An easy introduction to Pytorch for Neural Networks
subtitle:
version: 1.0
type: module
keywords: [PyTorch, NN, AI]   # there are always some keywords!
description: |
remarks:
    - my version of v1
todo:
sources:
    - title: An easy introduction to Pytorch for Neural Networks
      link: https://towardsdatascience.com/an-easy-introduction-to-pytorch-for-neural-networks-3ea08516bff2
file:
    usage:
        interactive: True
        terminal: True
    name: gan_03.py
    path: ~/Projects/AIML/GAN/basic/
    date: 2022-03-11
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""
#%%
import torch, torchvision
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as transforms

#%%
x = torch.Tensor(3, 3)
print(x)

x = torch.rand(3, 3)
print(x)

x = torch.ones(3,3)
y = torch.ones(3,3) * 4
z = x + y
print(z)

x = torch.ones(3,3) * 5
y = x[:, :2]
print(y)


#%%

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)

        x = F.relu(self.conv2(x))
        x = F.relu(self.conv2(x))
        x = self.global_pool(x)

        x = x.view(-1, 64)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        x = F.log_softmax(x)

        return x

model = Net()

#%%
num_epochs = 10
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

#%%
dir(torchvision.datasets)

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
"""Once that data is loaded, we’ll pass it to a torch DataLoader
which just gets it ready to pass to the model
with a specific batch size and optional shuffling.
"""
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

len(train_loader)

#%%
from itertools import *

for images, labels in islice(test_loader, 3):
    print(images)
    print(labels)

images.shape    # torch.Size([32, 1, 28, 28])
images.size()   # torch.Size([32, 1, 28, 28])
images.size(0)  # 32
images.size(2)  # 28
images.size(4)  #! IndexError: Dimension out of range (expected to be in range of [-4, 3], but got 4)

#%%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.CrossEntropyLoss()

#%%
# Train the model
total_step = len(train_loader)

for epoch in range(num_epochs):

    for i, (images, labels) in enumerate(train_loader):

        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


#%%
# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():   #!!!
    correct = 0
    total = 0

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        probability, category = torch.max(outputs.data, 1)   # category is an index of class with highest probabiity; see torch.max(data, axis)
        total += labels.size(0)
        correct += (category == labels).sum().item()

    print('Accuracy of the network on the MNIST test images: {} %'.format(100 * correct / total))

#%%
images, labels = next(islice(test_loader, 1))
images.shape    # torch.Size([32, 1, 28, 28])
labels.shape    # torch.Size([32])

outputs = model(images)
outputs
outputs.shape   # torch.Size([32, 10])
outputs.data
probability, category = torch.max(outputs.data, 1)
# category is an index of class with highest probabiity; see torch.max(data, axis)
probability     # tensor([-2.0266e-06, ...., -3.8265e-05])
category
probability, category = torch.max(outputs, 1)
probability     # tensor([-2.0266e-06, ...., -3.8265e-05],  grad_fn=<MaxBackward0>)      ???


#%% Saving
torch.save(model.state_dict(), 'model.ckpt')  # saves state of the model dictionary

#%% Loading Models
"""
The process for loading a model includes re-creating the model structure
and loading the state dictionary into it.
"""
model2 = Net()
model2.load_state_dict(torch.load("model.ckpt"))

#%%
