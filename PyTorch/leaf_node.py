"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: What is PyTorch leaf node?
version: 1.0
type: module
keywords: [pytorch, computation graph, visulisation, forward/backward pass]
description: |
remarks:
todo:
sources:
    - title: What is PyTorch leaf node?
      link: http://www.bnikolic.co.uk/blog/pytorch/python/2021/03/15/pytorch-leaf.html
    - title: PyTorchViz
      link: https://github.com/szagoruyko/pytorchviz
      description: A small package to create visualizations of PyTorch execution graphs
file:
    usage:
        interactive: True
        terminal: False
    name: leaf_node.py
    path: ~/Projects/AIML/NNRL/PytTorch/
    date: 2022-03-27
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""

#%%
import torch
import torch.nn as nn
import torchviz as tv

#%%
"""
[leaf_nodes_1.png]
In PyTorch  leaf nodes  are therefore the values from which the computation begins.
Here a simple program illustrating this:
"""

# The following two values are the leaf nodes
x=torch.ones(10, requires_grad=True)
y=torch.ones(10, requires_grad=True)

# The remaining nodes are not leaves:
def H(z1, z2):
    return torch.sin(z1**3) * torch.cos(z2**2)

def G(z1, z2):
    return torch.exp(z1) + torch.log(z2)

def F(z1, z2):
    return z1**3 * z2**0.5

h = H(x,y)
g = G(x,y)
f = F(h,g)
f

#%%
s = f.sum()
s       # tensor(1.5494, grad_fn=<SumBackward0>)

"""
The graph illustrating this is below
– the leaf nodes are the two blue nodes (i.e., our x and y above)
at the start of the computational graph:
"""
tv.make_dot(s)

#%%


#%%
