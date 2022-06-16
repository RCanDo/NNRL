#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Generative Adversarial Networks: Build Your First Models
subtitle: Handwritten Digits Generator With a GAN
version: 1.0
type: module
keywords: [GAN]   # there are always some keywords!
description: |
remarks:
    - my version of v1
todo:
sources:
    - title: Generative Adversarial Networks: Build Your First Models
      chapter: Handwritten Digits Generator With a GAN
      link: https://realpython.com/generative-adversarial-networks/
      date:
      authors:
          - nick:
            fullname:
            email:
      usage: |
          idea & inspiration
file:
    usage:
        interactive: True
        terminal: True
    name: gan_03.py
    path: ~/Projects/AIML/GAN/basic/
    date: 2022-03-08
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

import math
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

torch.manual_seed(111)

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#%%
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
"""
transforms.Normalize() changes the range of the coefficients to -1 to 1
by subtracting 0.5 from the original coefficients and dividing the result by 0.5.
...
The arguments of transforms.Normalize() are two tuples, (M₁, ..., Mₙ) and (S₁, ..., Sₙ),
with n representing the number of channels of the images.
Grayscale images such as those in MNIST dataset have only one channel, so the tuples have only one value.
Then, for each channel i of the image,
transforms.Normalize() subtracts Mᵢ from the coefficients and divides the result by Sᵢ.
"""

#%%
train_set = torchvision.datasets.MNIST(
    root=".", train=True, download=True, transform=transform
    )
#%%
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
    )

#%%
real_samples, mnist_labels = next(iter(train_loader))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])

#%%
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), 784)
        output = self.model(x)
        return output

"""
The vectorization occurs in the first line of .forward(),
as the call to x.view() converts the shape of the input tensor.
In this case, the original shape of the input x is 32 × 1 × 28 × 28,
where 32 is the batch size you’ve set up.
After the conversion, the shape of x becomes 32 × 784,
with each line representing the coefficients of an image of the training set.
"""

#%%
"""
Since the generator is going to generate more complex data,
it’s necessary to increase the dimensions of the input from the latent space.
In this case, the generator is going to be fed a 100-dimensional input
and will provide an output with 784 coefficients,
which will be organized in a 28 × 28 tensor representing an image.
"""
class Generator(nn.Module):

    def __init__(self, latent_dimension=100):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dimension, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        output = self.model(x)
        output = output.view(x.size(0), 1, 28, 28)
        return output

#%%
discriminator = Discriminator().to(device=device)

latent_dimension = 100
generator = Generator(latent_dimension).to(device=device)

#%%
lr = 0.0001
num_epochs = 50
loss_function = nn.BCELoss()

optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)


#%%
for epoch in range(num_epochs):

    for n, (real_samples, mnist_labels) in enumerate(train_loader):

        # Data for training the discriminator

        real_samples = real_samples.to(device=device)
        real_samples_labels = torch.ones((batch_size, 1)).to(device)

        latent_space_samples = torch.randn((batch_size, 100)).to(device)

        generated_samples = generator(latent_space_samples)
        generated_samples_labels = torch.zeros((batch_size, 1)).to(device)

        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat( (real_samples_labels, generated_samples_labels) )


        # Training the discriminator

        discriminator.zero_grad()

        output_discriminator = discriminator(all_samples)

        loss_discriminator = loss_function(

            output_discriminator, all_samples_labels

        )

        loss_discriminator.backward()

        optimizer_discriminator.step()


        # Data for training the generator

        latent_space_samples = torch.randn((batch_size, 100)).to(device)


        # Training the generator

        generator.zero_grad()

        generated_samples = generator(latent_space_samples)

        output_discriminator_generated = discriminator(generated_samples)

        loss_generator = loss_function(

            output_discriminator_generated, real_samples_labels

        )

        loss_generator.backward()

        optimizer_generator.step()


        # Show loss

        if n == batch_size - 1:

            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")

            print(f"Epoch: {epoch} Loss G.: {loss_generator}")

#%%
latent_space_samples = torch.randn(batch_size, 100).to(device)
generated_samples = generator(latent_space_samples)

#%%
generated_samples = generated_samples.cpu().detach()
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])

#%%



#%%
