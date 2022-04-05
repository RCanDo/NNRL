"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: a
version: 1.0
type: module
keywords: [pytorch, computation graph, forward/backward pass]
description: |
remarks:
todo:
sources:
    - title: PyTorch .detach() method
      link: http://www.bnikolic.co.uk/blog/pytorch-detach.html
    - title: 5 gradient/derivative related PyTorch functions
      link: https://attyuttam.medium.com/5-gradient-derivative-related-pytorch-functions-8fd0e02f13c6
file:
    usage:
        interactive: True
        terminal: False
    name: autograd_2.py
    path: ~/Projects/AIML/NNRL/PytTorch/
    date: 2022-03-29
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
"""PyTorch .detach() method
http://www.bnikolic.co.uk/blog/pytorch-detach.html
"""
x=torch.ones(10, requires_grad=True)
y=x**2
z=x**3
r=(y+z).sum()

tv.make_dot(r)
# tv.make_dot(r).render("attached", format="png")
tv.make_dot(r, show_attrs=True, show_saved=True)

r.backward()
x.grad          # tensor([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])

#%%
y=x**2
z=x.detach()**3
r=(y+z).sum()
tv.make_dot(r)

#%%
r.backward()
x.grad          # tensor([7., 7., 7., 7., 7., 7., 7., 7., 7., 7.])   WRONG !!!

# gradients are remembered and summed up

#%% we need to run everything from scratch !
del x, y, r

x=torch.ones(10, requires_grad=True)

y=x**2
z=x.detach()**3

r=(y+z).sum()

tv.make_dot(r)

r.backward()
x.grad          # tensor([2., 2., 2., 2., 2., 2., 2., 2., 2., 2.])   OK !!!


#%%


#%%
"""5 gradient/derivative related PyTorch functions
https://attyuttam.medium.com/5-gradient-derivative-related-pytorch-functions-8fd0e02f13c6
"""
#%%  .detach()   again
a = torch.arange(5., requires_grad=True)

b = a**2
c = a.detach()      # a already used;  error with .detach(), wrong result with .data

c.zero_()           # tensor([0., 0., 0., 0., 0.])      #!!!  all OK if you don't run this !!!    ??????????????

b.sum().backward()
"""!!!
! RuntimeError: one of the variables needed for gradient computation has been modified
! by an __inplace operation__: [torch.FloatTensor [5]] is at version 1;
  expected version 0 instead.
Hint: enable anomaly detection to find the operation that failed to compute its gradient,
      with
"""
torch.autograd.set_detect_anomaly(True)

"""see  autograd.py : In-place operations with autograd
or
https://pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd
"""
a.grad              # None
"""
It is because .detach() doesnt implicitly create a  _copy of the tensor_,
so when the tensor is modified later,
it’s updating the tensor _on the upstream side_ of .detach() too.
By cloning first, this issue doesnt arise, and all is ok.
"""
#%%  use .clone().detach()
del a, b, c

a = torch.arange(5., requires_grad=True)

b = a**2
c = a.clone().detach()      #!!!  .clone()

c.zero_()           # tensor([0., 0., 0., 0., 0.])

b.sum().backward()
a.grad              # tensor([0., 2., 4., 6., 8.])
"""
.detach() doesn’t create copies     !!!
it prevents the gradient to be computed, however, it still shares the data.

Thus, we should use .detach() when we don’t want to include a tensor in the resulting computational graph.
"""
help(c.zero_)

#%% no_grad()

#without using no_grad
x = torch.ones(3, requires_grad=True)
y = x**2
z = x**3
r = (y+z).sum()
print(r.requires_grad)      # True

#when using no_grad
x = torch.ones(3, requires_grad=True)
with torch.no_grad():
    y = x**2
    z = x**3
    r = (y+z).sum()
print(r.requires_grad)      # False

# obviously with no_grad() we cannot compute gradients:
x = torch.ones(3, requires_grad=True)
with torch.no_grad():
    y = x**2
    z = x**3
    r = (y+z).sum()
    r.backward()
#! RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn


#%% tensor.clone(memory_format=torch.preserve_format ) → Tensor
"""
tensor.clone() creates a copy of tensor that imitates the original tensor’s requires_grad field.

tensor.clone() maintains the connection with the computation graph.  !!!
That means, if you use the new cloned tensor, and derive the loss from the new one,
the gradients of that loss can be computed all the way back even beyond the point where the new tensor was created.

If you want to copy a tensor and detach from the computation graph you should be using

tensor.clone().detach()
"""
x=torch.ones(10, requires_grad=True)
x_clone = x.clone()
y=x_clone**2
z=x_clone**3

r=(y+z).sum()
r.backward()

print(x.grad)   # tensor([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])
print(x_clone.grad)     # None
""" UserWarning:
The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.
Its .grad attribute won't be populated during autograd.backward().
If you indeed want the .grad field to be populated for a non-leaf Tensor,
use  .retain_grad()  on the non-leaf Tensor.                                    !!!
If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead.
See github.com/pytorch/pytorch/pull/30531 for more informations.
(Triggered internally at  /opt/conda/conda-bld/pytorch_1648016052946/work/build/aten/src/ATen/core/TensorBody.h:412.)
  return self._grad
"""

"""
It should be noted that although the variables y and z are obtained using x_clone,
when r.backward() is done,
the gradients are propagated to the original tensor.                            !!!
"""
tv.make_dot(r)

#%%
del x, y, z
x=torch.ones(10, requires_grad=True)
x_clone = x.clone()
y=x**2
z=x_clone**3

r=(y+z).sum()
r.backward()

print(x.grad)           # tensor([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])
print(x_clone.grad)     # None
""" UserWarning: ...
"""

tv.make_dot(r)


#%% tensor.backward(gradient=None, retain_graph=None, create_graph=False)
"""
Computes the gradient of current tensor w.r.t. graph leaves.

The graph is differentiated using the chain rule.
If the tensor is non-scalar (i.e. its data has more than one element)
and requires gradient,
the function additionally requires specifying gradient.     ???
It should be a tensor of matching type and location,
that contains the gradient of the differentiated function w.r.t. self.      ???

This function accumulates gradients in the leaves           !!!
— you might need to zero them before calling it.

Parameters
gradient (Tensor or None) — Gradient w.r.t. the tensor.     ???
If it is a tensor, it will be automatically converted
to a Tensor that _does not require grad_ unless `create_graph` is True.
`None` values can be specified for scalar Tensors or ones that don’t require grad.
If a `None` value would be acceptable then this argument is optional.

retain_graph (bool, optional) — If False, the graph used to compute the grads
will be freed.
Note that in nearly all cases setting this option to True is not needed
and often can be worked around in a much more efficient way.
Defaults to the value of `create_graph`.

create_graph (bool, optional) — If True, graph of the derivative will be constructed,
allowing to compute higher order derivative products. Defaults to False.
"""
x=torch.ones(10, requires_grad = True )

y=x**2
z=x**3

r=(y+z).sum()
r.backward()
print(x.grad)       #  tensor([5., 5., 5., 5., 5., 5., 5., 5., 5., 5.])

tv.make_dot(r)

#%%
x=torch.ones(10, requires_grad = False )   # default !

y=x**2
z=x**3

r=(y+z).sum()
r.backward()        #! RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

tv.make_dot(r)      #!!!  only leaf in the graph...
"""
Thus, to use .backward() fuction on a variable,
we need to ensure that the leaf nodes that are involved must have
`requires_grad` set to True.

By default, pytorch expects .backward() to be called for the last output of the network
— the loss function.
The loss function always outputs a  _scalar_  and therefore,
the gradients of the scalar loss w.r.t all other variables/parameters is well defined (using the chain rule).
"""
#%%
v = torch.arange(1, 4, requires_grad=True, dtype=float)
w = 3*v**2 + torch.ones(3)
w.backward()                    # RuntimeError: grad can be implicitly created only for scalar outputs

w.backward(torch.tensor([1., 1., 1.]))
v.grad      # tensor([ 6., 12., 18.], dtype=torch.float64)

tv.make_dot(w)

w.grad
"""UserWarning:
The .grad attribute of a Tensor that is not a leaf Tensor is being accessed.
Its .grad attribute won't be populated during  autograd.backward().
If you indeed want the .grad field to be populated for a  non-leaf  Tensor,
use  .retain_grad()  on the  non-leaf  Tensor.
If you access the non-leaf Tensor by mistake, make sure you access the leaf Tensor instead.
See github.com/pytorch/pytorch/pull/30531 for more informations.
"""
#%%
del v, w
v = torch.arange(1, 4, requires_grad=True, dtype=float)
w = 3*v**2 + torch.ones(3)
w.backward(torch.tensor([1., 2., -1.]))
v.grad      # tensor([  6.,  24., -18.], dtype=torch.float64)

#%%
del v, w
v = torch.arange(1, 4, requires_grad=True, dtype=float)
w = 3*v**2 + torch.ones(3)
z = w.sum()
z.backward()                    # RuntimeError: grad can be implicitly created only for scalar outputs

tv.make_dot(z)
v.grad      # tensor([ 6., 12., 18.], dtype=torch.float64)

#%% 5. tensor.register_hook(hook)
"""
Registers a backward hook.

The hook will be called every time a gradient with respect to the Tensor is computed.
The hook should have the following signature:
    hook(grad) -> Tensor or None
The hook should not modify its argument,
but it can optionally return a new gradient which will be used in place of grad.

This function returns a handle with a method
    handle.remove()
that removes the hook from the module.
Use torch.Tensor.register_hook() directly on a specific input or output to get the required gradients.
"""
v = torch.tensor([0., 0., 0.], requires_grad = True )
h = v.register_hook(lambda grad: grad * 2)      # double the gradient;  h is a handle
v.backward(torch.tensor([1., 2., 3.]))
print(v.grad)       # tensor([2., 4., 6.])
print(h)            # torch.utils.hooks.RemovableHandle
h.remove()

#%%
v = torch.tensor([0., 0., 0.])
h = v.register_hook(lambda grad: grad * 2)  #! RuntimeError: cannot register a hook on a tensor that doesn't require gradient

#%%
