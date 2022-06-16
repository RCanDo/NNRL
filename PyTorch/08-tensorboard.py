#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: TensorBoard
subtitle:
version: 1.0
type: module
keywords: [TensorBoard, PyTorch, ]      # there are always some keywords!
description: |
remarks:
todo:
sources:
    - title: Visualizing Models, Data, and Training with TensorBoard
      link: https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
    - title:
      link:
file:
    usage:
        interactive: True
        terminal: True
    name: 08-tensorboard.py
    path: ~/Projects/AIML/NNRL/PyTorch/
    date: 2022-03-19
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""
#%%
# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# datasets
trainset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=True,
    transform=transform)
testset = torchvision.datasets.FashionMNIST('./data',
    download=True,
    train=False,
    transform=transform)

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                        shuffle=True, num_workers=2)


testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                        shuffle=False, num_workers=2)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

# helper function to show an image
# (used in the `plot_classes_preds` function below)
def matplotlib_imshow(img, one_channel=False):
    """
    img : tensor
    """
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

#%%
# for
# f(img) = Conv2d(in_channels = C, out_channels = K,  # K - nr of filters
#                 kernel_size = F, stride = S, padding = P, ...)(img)
# img.shape = (C, W, H)  (channels, width, height)
# f(img) -> img2;
# img2.shape = (K, W2, H2)
#   W2 = (W + 2P - F)/S + 1
#   H2 = (H + 2P - F)/S + 1
#


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)     # -> (6, 24, 24),   24 = (28 + 2*0 - 5)/1 + 1
        self.pool = nn.MaxPool2d(2, 2)      # -> (6, 12, 12)
        self.conv2 = nn.Conv2d(6, 16, 5)    # -> (16, 8, 8),     8 = (12 - 5) + 1
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))    # conv -> (6, 24, 24),   24 = (28 + 2*0 - 5)/1 + 1
                                                # pool -> (6, 12, 12)
        x = self.pool(F.relu(self.conv2(x)))    # conv -> (16, 8, 8),     8 = (12 + 2*0 - 5)/1 + 1
                                                # pool -> (16, 4, 4)
        x = x.view(-1, 16 * 4 * 4)      # -> ( ., 16 * 4 * 4)
        x = F.relu(self.fc1(x))         # -> ( ., 120)
        x = F.relu(self.fc2(x))         # -> ( ., 84)
        x = self.fc3(x)                 # -> ( ., 10)
        return x

net = Net()

#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

#%% 1. TensorBoard setup
"""
SummaryWriter is a key object for writing information to TensorBoard.
"""
# from torch.utils.tensorboard import SummaryWriter    # pytorch 1.11
# or
#
from tensorboardX import SummaryWriter     # for older pytorch and if tensorboardX properly installed
# else remove tensorboardX and install "oridnary" tensorboard
# from tensorboard.summary import Writer

# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/fashion_mnist_experiment_1')
#writer = Writer('runs/fashion_mnist_experiment_1')

"""
# in current folder:
tensorboardX --logdir runs
# or
tensorboard --logdir runs
"""
#%% 2. Writing to TensorBoard
"""
Now let’s write an image to our TensorBoard - specifically, a grid - using make_grid.
"""
# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()
images.shape    # torch.Size([4, 1, 28, 28])  (4 images, 1 channel, width, height)

# create grid of images
img_grid = torchvision.utils.make_grid(images)
img_grid.shape  # torch.Size([3, 32, 122])   32 = 28 + 2*2,  122 = 4*28 + 5*2

# show images
matplotlib_imshow(img_grid, one_channel=False)
matplotlib_imshow(img_grid, one_channel=True)

# write to tensorboard
writer.add_image('four_fashion_mnist_images', img_grid)

#%% 3. Inspect the model using TensorBoard
writer.add_graph(net, images)
writer.close()

#%% 4. Adding a “Projector” to TensorBoard
"""
We can visualize the lower dimensional representation of higher dimensional data via the add_embedding method
"""
# helper function
def select_n_random(data, labels, n=100):
    '''
    Selects n random datapoints and their corresponding labels from a dataset
    '''
    assert len(data) == len(labels)

    perm = torch.randperm(len(data))
    return data[perm][:n], labels[perm][:n]

# select random images and their target indices
images, labels = select_n_random(trainset.data, trainset.targets)

# get the class labels for each image
class_labels = [classes[lab] for lab in labels]

# log embeddings
features = images.view(-1, 28 * 28)
writer.add_embedding(features,
                    metadata=class_labels,
                    label_img=images.unsqueeze(1))
writer.close()

#%% 5. Tracking model training with TensorBoard

# helper functions

def images_to_probs(net, images):
    '''
    Generates predictions and corresponding probabilities from a trained
    network and a list of images
    '''
    output = net(images)
    # convert output probabilities to predicted class
    _, preds_tensor = torch.max(output, 1)
    preds = np.squeeze(preds_tensor.numpy())
    return preds, [F.softmax(el, dim=0)[i].item() for i, el in zip(preds, output)]


def plot_classes_preds(net, images, labels):
    '''
    Generates matplotlib Figure using a trained network, along with images
    and labels from a batch, that shows the network's top prediction along
    with its probability, alongside the actual label, coloring this
    information based on whether the prediction was correct or not.
    Uses the "images_to_probs" function.
    '''
    preds, probs = images_to_probs(net, images)
    # plot the images in the batch, along with predicted and true labels
    fig = plt.figure(figsize=(12, 48))
    for idx in np.arange(4):
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])
        matplotlib_imshow(images[idx], one_channel=True)
        ax.set_title("{0}, {1:.1f}%\n(label: {2})".format(
            classes[preds[idx]],
            probs[idx] * 100.0,
            classes[labels[idx]]),
                    color=("green" if preds[idx]==labels[idx].item() else "red"))
    return fig


#%%
running_loss = 0.0
for epoch in range(1):  # loop over the dataset multiple times

    for i, data in enumerate(trainloader, 0):

        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:    # every 1000 mini-batches...

            # ...log the running loss
            writer.add_scalar('training loss',
                            running_loss / 1000,
                            epoch * len(trainloader) + i)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(net, inputs, labels),
                            global_step=epoch * len(trainloader) + i)
            running_loss = 0.0
print('Finished Training')


#%%
