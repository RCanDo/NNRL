#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: PyTorchViz
version: 1.0
type: module
keywords: [pytorch, computation graph, visulisation, forward/backward pass]
description: |
remarks:
todo:
sources:
    - title: PyTorchViz
      link: https://github.com/szagoruyko/pytorchviz
      description: A small package to create visualizations of PyTorch execution graphs
    - title: PyTorchViz examples
      link: https://colab.research.google.com/github/szagoruyko/pytorchviz/blob/master/examples.ipynb#scrollTo=spWKUcGvPdGv
file:
    usage:
        interactive: True
        terminal: True
    name: torchviz.py
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
dir(tv)     # dot, make_dot, make_dot_from_trace
help(tv.dot)
"""
get_fn_name(fn, show_attrs, max_attr_chars)

make_dot(var, params=None, show_attrs=False, show_saved=False, max_attr_chars=50)

    Produces Graphviz representation of PyTorch autograd graph.

    If a node represents a _backward function_, it is gray.

    Otherwise, the node represents a tensor and is either blue, orange, or green:
     - Blue: reachable leaf tensors that requires grad (tensors whose `.grad`
         fields will be populated during `.backward()`)
     - Orange: saved tensors of custom autograd functions as well as those
         saved by built-in backward nodes
     - Green: tensor passed in as outputs
     - Dark green: if any output is a view, we represent its base tensor with
         a dark green node.

    Args:
        var: output tensor
        params: dict of (name, tensor) to add names to node that requires grad
        show_attrs: whether to display non-tensor attributes of backward nodes
            (Requires PyTorch version >= 1.9)
        show_saved: whether to display saved tensor nodes that are not by custom
            autograd functions. Saved tensor nodes for custom functions, if
            present, are always displayed. (Requires PyTorch version >= 1.9)
        max_attr_chars: if show_attrs is `True`, sets max number of characters
            to display for any given attribute.

make_dot_from_trace(trace)
    This functionality is not available in pytorch core at
    https://pytorch.org/docs/stable/tensorboard.html

resize_graph(dot, size_per_element=0.15, min_size=12)
    Resize the graph according to how much content it contains.

    Modify the graph in place.
"""
#%%
model = nn.Sequential()
model.add_module('W0', nn.Linear(8, 16))
model.add_module('tanh', nn.Tanh())
model.add_module('W1', nn.Linear(16, 1))

x = torch.randn(1, 8)
y = model(x)

tv.make_dot(y.mean(), params=dict(model.named_parameters()))

# Set show_attrs=True and show_saved=True to see what autograd saves for the backward pass.
# (Note that this is only available for pytorch >= 1.9.)
tv.make_dot(y.mean(), params=dict(model.named_parameters()), show_attrs=True, show_saved=True)

#%%
