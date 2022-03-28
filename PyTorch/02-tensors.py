#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Tensors
subtitle:
version: 1.0
type: module
keywords: [PyTorch, NN, AI]   # there are always some keywords!
description: |
remarks:
    - my version of v1
todo:
sources:
    - title: Tensors
      link: https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html
file:
    usage:
        interactive: True
        terminal: True
    name: 02-tensors.py
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
import torch
import numpy as np

# from torch import nn
# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

#%%
"""
Tensors are similar to NumPy’s ndarrays,
except that tensors can run on GPUs or other hardware accelerators.
In fact, tensors and NumPy arrays can often share the same underlying memory,
eliminating the need to copy data (see 'Bridge with NumPy' below).
Tensors are also optimized for automatic differentiation
(we’ll see more about that later in the 'Autograd' section).
If you’re familiar with ndarrays, you’ll be right at home with the Tensor API.
If not, follow along!
"""
#%% Directly from data
# Tensors can be created directly from data. The data type is automatically inferred.

data = [[1, 2, 3],[4, 5, 6]]
x_data = torch.tensor(data)
x_data

#%% From a NumPy array
# Tensors can be created from NumPy arrays (and vice versa - see Bridge with NumPy).

np_array = np.array(data)
x_np = torch.from_numpy(np_array)
x_np    # tensor([...])

#%% From another tensor:
# The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.

x_ones = torch.ones_like(x_data) # retains the properties of x_data
x_ones

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
x_rand

x_zeros = torch.zeros_like(x_data, dtype=torch.float) # overrides the datatype of x_data
x_zeros

#%% With random or constant values:
# shape is a tuple of tensor dimensions.
# In the functions below, it determines the dimensionality of the output tensor.

shape = (2,3,)

rand_tensor = torch.rand(shape)
rand_tensor

ones_tensor = torch.ones(shape)
ones_tensor

zeros_tensor = torch.zeros(shape)
zeros_tensor

#%% Attributes of a Tensor
# Tensor attributes describe their shape, datatype, and the device on which they are stored.

tensor = torch.rand(3,4)
dir(tensor)     #! a lot !
type(tensor)    # torch.Tensor
tensor.dtype    # torch.float32
tensor.shape    # torch.Size([3, 4])

tensor.size(0)
tensor.size(1)

tensor.device   # device(type='cpu')

#%% Operations on Tensors
"""
Over 100 tensor operations, including arithmetic, linear algebra,
matrix manipulation (transposing, indexing, slicing), sampling and more
are comprehensively described [here](https://pytorch.org/docs/stable/torch.html).

Each of these operations can be run on the GPU (at typically higher speeds than on a CPU).
If you’re using Colab, allocate a GPU by going to Runtime > Change runtime type > GPU.

By default, tensors are created on the CPU.
We need to explicitly move tensors to the GPU using .to() method (after checking for GPU availability).
Keep in mind that copying large tensors across devices can be expensive in terms of time and memory!
"""
# We move our tensor to the GPU if available
if torch.cuda.is_available():
    tensor = tensor.to("cuda")

#%%
torch.manual_seed(222)
tensor = torch.randint(-3, 4, (2, 3))
tensor
tensor[0]
tensor[:, 0]
tensor[..., -1]
tensor[-1, ...]
tensor[-1, :]
tensor[:, 1] = 0
tensor

#%%
"""
Joining tensors You can use torch.cat() to concatenate a sequence of tensors
along a given dimension.
See also torch.stack(), another tensor joining op that is subtly different from torch.cat.
"""
t1 = torch.cat([x_ones, x_rand, x_zeros], dim=1)
#! RuntimeError: Expected object of scalar type Long but got scalar type Float for sequence element 1 in sequence argument at position #1 'tensors'
x_ones.dtype    # torch.ones_like
x_rand.dtype    # torch.float32
x_zeros.dtype   # torch.float32

help(torch.cat)
# ... any python sequence of tensors of the same type. !!!

x_ones.type(torch.float32)
t1 = torch.cat([x_ones.type(torch.float32), x_rand, x_zeros], dim=1)
t1

#%% Arithmetic operations

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
tensor
y1 = tensor @ tensor.T
y1
y2 = tensor.matmul(tensor.T)
y2

y3 = torch.rand_like(tensor)   #! RuntimeError: _th_uniform_ not supported on CPUType for Long
y3 = torch.rand_like(tensor.type(torch.float32))
y3

torch.matmul(tensor, tensor.T)
torch.matmul(tensor, tensor.T, out=y3)  #! RuntimeError: Expected object of scalar type Long but got scalar type Float for argument #0 'result' in call to _th_mm_out

tensor32 = tensor.type(torch.float32)
torch.matmul(tensor32, tensor32.T, out=y3)
y3

# This computes the  element-wise  product. z1, z2, z3 will have the same value
z1 = tensor * tensor
z1
z2 = tensor.mul(tensor)
z2
z3 = torch.rand_like(tensor)    #! RuntimeError: _th_uniform_ not supported on CPUType for Long
z3 = torch.rand_like(tensor.type(torch.float32))
z3
torch.mul(tensor, tensor, out=z3)
z3

#%% Single-element tensors
"""
If you have a one-element tensor, for example by aggregating all values of a tensor into one value,
you can convert it to a Python numerical value using item():
"""
agg = tensor.sum()
agg   # tensor
agg_item = agg.item()
agg_item   # float

#%% In-place operations
"""Operations that store the result into the operand are called in-place.
They are denoted by a _ suffix. For example: x.copy_(y), x.t_(), will change x.
"""
tensor
tensor.add_(5)
tensor
tensor.sub_(2.)  # RuntimeError: result type Float can't be cast to the desired output type Long
tensor.sub_(2)
tensor

# so it also returns !!!
t0 = tensor.add_(1)
tensor
t0

#%% Bridge with NumPy
"""
Tensors on the CPU and NumPy arrays can share their underlying memory locations,
and changing one will change the other.
"""
# Tensor to NumPy array

t = torch.ones(5)
t       # tensor([1., 1., 1., 1., 1.])
n = t.numpy()
n       # array([1., 1., 1., 1., 1.], dtype=float32)

#%% NumPy array to Tensor
n = np.ones(5)
n       # array([1., 1., 1., 1., 1.])
t = torch.from_numpy(n)
t       # tensor([1., 1., 1., 1., 1.], dtype=torch.float64)

#!!! Changes in the NumPy array reflects in the tensor.

np.add(n, 1, out=n)
n       # array([2., 2., 2., 2., 2.])
t       # tensor([2., 2., 2., 2., 2.], dtype=torch.float64)

#%%
