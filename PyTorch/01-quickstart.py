#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Quickstart
subtitle:
version: 1.0
type: module
keywords: [PyTorch, NN, AI]   # there are always some keywords!
description: |
remarks:
    - my version of v1
todo:
sources:
    - title: Quickstart
      link: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
file:
    usage:
        interactive: True
        terminal: True
    name: 01-quickstart.py
    path: ~/Roboczy/Python/PyTorch/
    date: 2022-03-13
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""

#%%
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#%%
dir(datasets)

"""Every TorchVision Dataset includes two arguments:
`transform` and `target_transform`
to modify the samples and labels respectively.
"""

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,     #!
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,    #!
    download=True,
    transform=ToTensor(),
)

dir(training_data)
training_data.data.shape
type(training_data)         # torchvision.datasets.mnist.FashionMNIST
type(training_data.data)    # torch.Tensor
training_data.targets       # tensor([...])
training_data.targets.shape # torch.Size([60000])
training_data.classes
# ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

training_data.train_labels  #! warnings.warn("train_labels has been renamed targets")
training_data.test_labels   #! warnings.warn("test_labels has been renamed targets"
all(training_data.test_labels == training_data.train_labels)  # True !!!

training_data.target_transform  # None

#%%
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#%% the same by itertools.islice()
from itertools import islice
images, labels = next(islice(test_dataloader, 1))
images.shape    # torch.Size([64, 1, 28, 28])
labels.shape    # torch.Size([64])


#%% Creating Models
#
"""
To define a neural network in PyTorch,
we create a class that inherits from nn.Module.
We define the layers of the network in the __init__ function
and specify how data will pass through the network in the forward function.
To accelerate operations in the neural network, we move it to the GPU if available.

"""

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):

    def __init__(self):

        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

#%% Optimizing the Model Parameters
"""
To train a model, we need a  _loss function_  and an  _optimizer_.
"""
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD( model.parameters(), lr=1e-3 )

def train(dataloader, model, loss_fn, optimizer):
    """"""
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

#%%
def test(dataloader, model, loss_fn):
    """"""
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()                #???
    test_loss, correct = 0, 0

    with torch.no_grad():       #!!!
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()    #???
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#%%
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")

#%%
classes = test_data.classes
classes
# ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')


#%% Saving Models
torch.save(model.state_dict(), "quickstart_model.pth")
print("Saved PyTorch Model State to `quickstart_model.pth`")

#%% Loading Models
"""
The process for loading a model includes re-creating the model structure
and loading the state dictionary into it.
"""
model2 = NeuralNetwork()
model2.load_state_dict(torch.load("quickstart_model.pth"))

#%%
