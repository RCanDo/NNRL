"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Autograd mechanics
version: 1.0
type: module
keywords: [pytorch, computation graph, forward/backward pass]
description: |
remarks:
todo:
sources:
    - title: Autograd mechanics
      link: https://pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc
    - title: PyTorchViz
      link: https://github.com/szagoruyko/pytorchviz
      description: A small package to create visualizations of PyTorch execution graphs
file:
    usage:
        interactive: True
        terminal: True
    name: autograd_mechanics.py
    path: ~/Projects/AIML/NNRL/PytTorch/
    date: 2022-03-28
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""
#%%
"""
PyTorch 1.11

...
An important thing to note is that the graph is recreated from scratch at every iteration,
and this is exactly what allows for using arbitrary Python control flow statements,
that can change the overall shape and size of the graph at every iteration.
You don’t have to encode all possible paths before you launch the training
- WHAT YOU RUN IS WHAT YOU DIFFERENTIATE.
...

"""
#%%
import torch
import torch.nn as nn
import torchviz as tv

#%%
x = torch.randn(5, requires_grad=True)
y = x.pow(2)
y.grad_fn       # PowBackward0
dir(y.grad_fn)
print(x.equal(y.grad_fn._saved_self))   # True
print(x is y.grad_fn._saved_self)       # True

tv.make_dot(y)

#%%
x = torch.randn(5, requires_grad=True)
y = x.exp()
print(y.equal(y.grad_fn._saved_result)) # True
print(y is y.grad_fn._saved_result)     # False

#%%
