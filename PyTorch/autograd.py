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
        terminal: False
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
x
y = x.pow(2)
y.grad_fn       # PowBackward0

dir(y.grad_fn)
y.grad_fn._saved_self   # == x

print(x.equal(y.grad_fn._saved_self))   # True
print(x is y.grad_fn._saved_self)       # True

tv.make_dot(y)

#%%
x = torch.randn(5, requires_grad=True)
y = x.exp()
print(y.equal(y.grad_fn._saved_result)) # True
print(y is y.grad_fn._saved_result)     # False

#%%


#%%
"""
...
Locally disabling gradient computation
https://pytorch.org/docs/stable/notes/autograd.html#locally-disabling-gradient-computation

1. Setting requires_grad

Grad Modes
2. Default Mode (Grad Mode)
3. No-grad Mode
4. Inference Mode
5. Evaluation Mode (nn.Module.eval())
"""

#%% no_grad()
# https://pytorch.org/docs/stable/generated/torch.autograd.no_grad.html#torch.autograd.no_grad
x = torch.tensor([1.], requires_grad=True)
with torch.no_grad():
  y = x * 2
y.requires_grad     # False

@torch.no_grad()
def doubler(x):
    return x * 2
z = doubler(x)
z.requires_grad     # False

#%% enable_grad()
# https://pytorch.org/docs/stable/generated/torch.autograd.enable_grad.html#torch.autograd.enable_grad
x = torch.tensor([1.], requires_grad=True)
with torch.no_grad():
  with torch.enable_grad():
    y = x * 2
y.requires_grad     # True

y.backward()
x.grad              # tensor([2.])

@torch.enable_grad()
def doubler(x):
    return x * 2
with torch.no_grad():
    z = doubler(x)
z.requires_grad     # False

#%% set_grad_enabled()
# https://pytorch.org/docs/stable/generated/torch.autograd.set_grad_enabled.html#torch.autograd.set_grad_enabled
# mode (bool) – Flag whether to enable grad (True), or disable (False).
# This can be used to conditionally enable gradients.

x = torch.tensor([1.], requires_grad=True)
is_train = False
with torch.set_grad_enabled(is_train):
  y = x * 2
y.requires_grad     # False

torch.set_grad_enabled(True)
y = x * 2
y.requires_grad     # True

torch.set_grad_enabled(False)
y = x * 2
y.requires_grad     # False

#%% inference_mode()    Torch 1.10
# https://pytorch.org/docs/stable/generated/torch.autograd.inference_mode.html#torch.autograd.inference_mode
# mode (bool) – Flag whether to enable or disable inference mode

x = torch.ones(1, 2, 3, requires_grad=True)
with torch.inference_mode():
  y = x * x
y.requires_grad     # False

y._version      #! RuntimeError: Inference tensors do not track version counter.

@torch.inference_mode()
def func(x):
  return x * x
out = func(x)
out.requires_grad   # False

#%%
"""
In-place operations with autograd
https://pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd

Supporting in-place operations in autograd is a hard matter,
and we discourage their use in most cases.

Autograd’s aggressive buffer freeing and reuse makes it very efficient
and there are very few occasions when in-place operations
actually lower memory usage by any significant amount.

Unless you’re operating under heavy memory pressure, you might never need to use them.

There are two main reasons that limit the applicability of in-place operations:

1. In-place operations can potentially _overwrite values_ required to compute gradients.

2. Every in-place operation actually requires the implementation to rewrite the computational graph.
   Out-of-place versions simply allocate new objects and keep references to the old graph,
   while in-place operations, require changing the creator of all inputs
   to the Function representing this operation.
   This can be tricky, especially if there are many Tensors that reference the same storage
   (e.g. created by indexing or transposing),
   and
   in-place functions will actually raise an error if the storage of modified inputs    !!!
   is referenced by any other Tensor.                                                   !!!


In-place correctness checks

Every tensor keeps a _version counter_                                                  !!!
that is incremented every time it is _marked dirty_ in any operation.
When a Function saves any tensors for backward,
a version counter of their containing Tensor is saved as well.
Once you access   self.saved_tensors
it is checked, and if it is greater than the saved value an error is raised.            !!!
This ensures that if you’re using in-place functions and not seeing any errors,
you can be sure that the computed gradients are correct.

...
"""
