#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Generative Adversarial Networks: Build Your First Models
subtitle: Your First GAN
version: 2.0
type: module
keywords: [GAN]   # there are always some keywords!
description: |
remarks:
    - my version of v1
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

torch.manual_seed(111)

#%%
N = 30 * 32        # real_xx_length
real_xx = torch.zeros((N, 2))
real_xx[:, 0] = 2 * math.pi * torch.rand(N)
real_xx[:, 1] = torch.sin(real_xx[:, 0])
real_labels = torch.ones(N)    # should be ones !
real_data = [ (real_xx[i], real_labels[i]) for i in range(N) ]

#%%
plt.plot(real_xx[:, 0], real_xx[:, 1], ".")

#%%
B = 32      # batch size
loader = torch.utils.data.DataLoader(real_data, batch_size=B, shuffle=True)

#%%
"""
for n, (real_batch, labels) in enumerate(loader):
    print(f"{n} : {real_batch} ; {labels}")
"""
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

class Generator(nn.Module):

    def __init__(self, latent_dimension=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dimension, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            )

    def forward(self, x):
        output = self.model(x)
        return output

#%%
latent_dimension = 2

discriminator = Discriminator()
generator = Generator(latent_dimension)

LR = 0.001      # Learning Rate
epochs = 300
loss_function = nn.BCELoss()        # Binary Cross Entropy
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=LR)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=LR)

#%% btw:
for p in discriminator.parameters(): print(p.shape)
print("---")
for p in generator.parameters(): print(p.shape)

#%% training loop

def learn(e, s, real_batch, real_labels):
    """"""
    latent_sample = torch.randn((B, latent_dimension))
    # generator()
    fake_sample = generator(latent_sample)     # at first completely random result
    fake_labels = torch.zeros((B, 1))

    both_samples = torch.cat((real_batch, fake_sample))

    real_labels = real_labels.reshape((-1, 1))
    #real_labels = torch.ones((B, 1))       # _ should be ones already
    both_labels = torch.cat((real_labels, fake_labels))   # length 64

    # Training the discriminator
    discriminator.zero_grad()
    # discriminator()
    both_labels_hat = discriminator(both_samples)

    loss_discriminator = loss_function(both_labels_hat, both_labels)
    loss_discriminator.backward()
    optimizer_discriminator.step()

    # Data for training the generator
    latent_sample = torch.randn((B, latent_dimension))   # again ? must be ... :( see (*) below

    # Training the generator
    generator.zero_grad()
    fake_sample = generator(latent_sample)   # again ?
    fake_labels_hat = discriminator(fake_sample)
    loss_generator = loss_function(fake_labels_hat, real_labels)
    loss_generator.backward()
    optimizer_generator.step()

    # Show loss
    if e % 10 == 0 and s == 30 - 1:
        print(f"Epoch: {e} Loss D.: {loss_discriminator}")
        print(f"Epoch: {e} Loss G.: {loss_generator}")
        #plt.close("all")
        #fake_sample = fake_sample.detach()
        #plt.clf()
        #plt.plot(real_xx[:, 0], real_xx[:, 1], ".")
        #plt.plot(fake_sample[:, 0], fake_sample[:, 1], ".")


for e in range(epochs):
    for s, (real_batch, real_labels) in enumerate(loader):
        learn(e, s, real_batch, real_labels)


latent_sample = torch.randn(100, latent_dimension)
fake_sample = generator(latent_sample)

"""
Before plotting the generated_samples data, you’ll need to use .detach()
to return a tensor from the PyTorch computational graph,
which you’ll then use to calculate the gradients:
"""
fake_sample = fake_sample.detach()
plt.plot(fake_sample[:, 0], fake_sample[:, 1], ".")

#%%
""" (*)
RuntimeError: Trying to backward through the graph a second time,
but the saved intermediate results have already been freed.
Specify retain_graph=True when calling .backward() or autograd.grad() the first time.
"""

#%%
#%%
latent_dimension = 1

discriminator = Discriminator()
generator = Generator(latent_dimension)

for e in range(epochs):
    for s, (real_batch, real_labels) in enumerate(loader):
        learn(e, s, real_batch, real_labels)


latent_sample = torch.randn(100, latent_dimension)
fake_sample = generator(latent_sample)

"""
Before plotting the generated_samples data, you’ll need to use .detach()
to return a tensor from the PyTorch computational graph,
which you’ll then use to calculate the gradients:
"""
fake_sample = fake_sample.detach()
plt.plot(fake_sample[:, 0], fake_sample[:, 1], ".")     ## bad...


#%%
#%%  Handwritten Digits Generator With a GAN
