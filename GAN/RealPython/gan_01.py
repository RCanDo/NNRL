#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Generative Adversarial Networks: Build Your First Models
subtitle: Your First GAN
version: 1.0
type: module
keywords: [GAN]   # there are always some keywords!
description: |
remarks:
todo:
sources:
    - title: Generative Adversarial Networks: Build Your First Models
      chapter: Your First GAN
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
    name: gan_01.py
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

#%%
#torch.manual_seed(111)   # 10622548165007232136

#%%
train_data_length = 1024
train_data = torch.zeros((train_data_length, 2))
train_data[:, 0] = 2 * math.pi * torch.rand(train_data_length)
train_data[:, 1] = torch.sin(train_data[:, 0])
train_labels = torch.zeros(train_data_length)   # should be ones !
train_set = [ (train_data[i], train_labels[i]) for i in range(train_data_length) ]

#%%
plt.plot(train_data[:, 0], train_data[:, 1], ".")

#%%
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
    )

#%%
for n, (real_samples, _) in enumerate(train_loader):
    print(f"{n} : {real_samples} ; {_}")

#%%
class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
            )

    def forward(self, x):
        output = self.model(x)
        return output

#%%
class Generator(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            )

    def forward(self, x):
        output = self.model(x)
        return output


#%%
discriminator = Discriminator()
generator = Generator()

#%%
lr = 0.001
epochs = 300
loss_function = nn.BCELoss()        # Binary Cross Entropy
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

#%% btw:
for p in discriminator.parameters(): print(p.shape)
for p in generator.parameters(): print(p.shape)

#%% training loop

for epoch in range(epochs):

    for n, (real_samples, _) in enumerate(train_loader):

        # Data for training the discriminator
        real_samples_labels = torch.ones((batch_size, 1))       # _ should be ones already

        latent_space_samples = torch.randn((batch_size, 2))
        # generator()
        generated_samples = generator(latent_space_samples)     # at first completely random result
        generated_samples_labels = torch.zeros((batch_size, 1))

        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat(
            (real_samples_labels, generated_samples_labels)
            )   # length 64

        # Training the discriminator
        discriminator.zero_grad()
        # discriminator()
        output_discriminator = discriminator(all_samples)

        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Data for training the generator
        latent_space_samples = torch.randn((batch_size, 2))

        # Training the generator
        generator.zero_grad()
        generated_samples = generator(latent_space_samples)
        output_discriminator_generated = discriminator(generated_samples)
        loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()

        # Show loss
        if epoch % 10 == 0 and n == batch_size - 1:         # 1024 = 32**2
            print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
            print(f"Epoch: {epoch} Loss G.: {loss_generator}")


#%%
latent_space_samples = torch.randn(100, 2)
generated_samples = generator(latent_space_samples)

#%%
"""
Before plotting the generated_samples data, you’ll need to use .detach()
to return a tensor from the PyTorch computational graph,
which you’ll then use to calculate the gradients:
"""
generated_samples = generated_samples.detach()
plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")

for p in generator.parameters(): print(p)


#%%
#%%  Handwritten Digits Generator With a GAN
