#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Loading Imagas With Torchvision
subtitle:
version: 1.0
type: module
keywords: [PyTorch, Torchvision, PIL, imageio]      # there are always some keywords!
description: |
remarks:
todo:
sources:
    - title: Pillow library
      link: ~/Roboczy/Python/graphics/Pillow/01-quick_guide.py
    - title: imageio library
      link:
file:
    usage:
        interactive: True
        terminal: True
    name: 01-loading_images.py
    path: ~/Projects/AIML/NNRL/PyTorch/torchvision/
    date: 2022-03-19
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""
#%%
#%%
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
pokemons = Path("/home/arek/Projects/AIML/contrib/gans/data/pokemon/")

#%%  . -> pillow            ! (96, 96)
from PIL import Image
pmon = pokemons / "10007.png"
im = Image.open(pmon)
type(im)    # PIL.PngImagePlugin.PngImageFile
im
im.show()
plt.imshow(im)
im.size         # (96, 96)

# !!! mode !!!  etc. see https://pillow.readthedocs.io/en/latest/handbook/concepts.html#modes
im.mode         # 'P'   not 'RGB' :(
im.format       # 'PNG'
im.getbands()   # ('P',)   !!! The palette mode (P) uses a color palette to define the actual color for each pixel.

im.palette      # <PIL.ImagePalette.ImagePalette at 0x7f09c6667190>
im.palette.colors
im.palette.mode     # 'RGB'
im.palette.tobytes()
im.palette.tostring()

#%%
im_rgba = im.convert('RGBA')
plt.imshow(im_rgba)
r, g, b, a = im_rgba.split()
a       # alpha layer
np.unique(a)    # array([  0, 255], dtype=uint8)
#  like a mask: 0 -- total transparence = no background, 255 -- not transparent at all

im_rgb = im.convert('RGB')
plt.imshow(im_rgb)
# black background

#%%  pillow -> tensor       !!!  torch.Size([ 1 , 96, 96])
import torch
from torchvision.transforms import ToTensor
tim = ToTensor()(im)
tim.shape           # torch.Size([1, 96, 96])
plt.imshow(tim)     #! TypeError: Invalid shape (1, 96, 96) for image data

plt.imshow(tim.permute(1, 2, 0))     #! TypeError: Invalid shape (96, 96, 1) for image data

#!!! HENCE it must be  (width, heigh, 4)  !!!  for plotting
# but   ()  for NN

#%%
im_rgb = im.convert('RGBA')
tim_rgb = ToTensor()(im_rgb)
tim_rgb.shape       #! torch.Size([4, 96, 96])

#%%  . -> tensor            ! RuntimeError: Non RGB images are not supported.
import torchvision
from torchvision.io import read_image
tim = read_image(pmon)          #! RuntimeError: image::read_file() Expected a value of type 'str' for argument '_0' but instead found type 'PosixPath'.
tim = read_image(str(pmon)) #! RuntimeError: Non RGB images are not supported.

#%%  . -> imageio           !!!! np.array([96, 96,  4 ])
import imageio as iio
iim = iio.imread(pmon)      # np.array
plt.imshow(iim)
iim.shape       # (96, 96, 4)

#%%  imageio -> tensor      !!!  torch.Size([ 4 , 96, 96])
tiim = ToTensor()(iim)
tiim.shape        # torch.Size([4, 96, 96])
plt.imshow(tiim)  # Invalid shape (4, 96, 96) for image data

tiim.permute(1, 2, 0).shape         # torch.Size([96, 96, 4])
plt.imshow(tiim.permute(1, 2, 0))   # OK

#%%  imageio -> pillow
from torchvision.transforms import ToPILImage
piim = ToPILImage()(iim)
piim
piim.size       # (96, 96)
piim.mode       # 'RGBA'

#%% imageio -> pillow -> tensor
tpiim = ToTensor()(piim)
tpiim.shape     # torch.Size([4, 96, 96])

#%%  tensor -> pillow       ! (96, 96)
ptiim = ToPILImage()(tiim)
ptiim
ptiim.size      # (96, 96)
ptiim.mode      # 'RGBA'

#%%
import numpy as np
from torchvision.utils import make_grid
plt.imshow(make_grid([torch.tensor(iim), tiim.permute(1, 2, 0)]))

grid = make_grid([torch.tensor(iim), tiim.permute(1, 2, 0)])
grid.shape      # torch.Size([96, 100, 14])

grid = make_grid([torch.tensor(iim), torch.tensor(iim)])
grid.shape      # torch.Size([96, 100, 14])


#%%
iim = iio.imread('imageio:astronaut.png')
iim.shape
plt.imshow(iim)

#%%
