#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begins with ---

title: Automatic Differentiation with torch.autograd
subtitle:
version: 1.0
type: module
keywords: [PyTorch, NN, AI]   # there are always some keywords!
description: |
remarks:
    - my version of v1
todo:
sources:
    - title: Automatic Differentiation with torch.autograd
      link: https://pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html
    - title: Autograd mechanics
      link: https://pytorch.org/docs/stable/notes/autograd.html
      description: |
          How autograd works in detail. Long!
    - title: Automatic differentiation package - torch.autograd
      link: https://pytorch.org/docs/stable/autograd.html#function
      description:
          torch.aotograd spec
file:
    usage:
        interactive: True
        terminal: True
    name: 06-gradients.py
    path: ~/Projects/AIML/NNRL/PyTorch
    date: 2022-03-17
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
              - arek@staart.pl
"""
#%%
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
xw = torch.matmul(x, w)
z = xw + b
z
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
loss

#%%
"""
You can set the value of requires_grad when creating a tensor,
or later by using  x.requires_grad_(True)  method.

A function that we apply to tensors to construct computational graph
is in fact an object of class  Function.
This object knows how to compute the function in the forward direction,
and also how to compute its derivative during the backward propagation step.

A reference to the backward propagation function is stored in  grad_fn  property of a tensor.
You can find more information of Function in the documentation:
https://pytorch.org/docs/stable/autograd.html#function
"""
w.grad_fn
print(f"Gradient function for w = {w.grad_fn}")
# Gradient function for w = None
print(f"Gradient function for b = {b.grad_fn}")
# Gradient function for b = None

print(f"Gradient function for xw = {xw.grad_fn}")
# Gradient function for xw = <SqueezeBackward3 object at 0x7f76303f78e0>
print(f"Gradient value for xw = {xw.grad}")     # None

print(f"Gradient function for z = {z.grad_fn}")
# Gradient function for z = <AddBackward0 object at 0x7f763023df10>

loss.grad_fn
print(f"Gradient function for loss = {loss.grad_fn}")
# Gradient function for loss = <BinaryCrossEntropyWithLogitsBackward object at 0x7f763023dc70>

#%%
"""
To optimize weights of parameters in the neural network,
we need to compute the derivatives of our loss function with respect to parameters,
namely, we need
∂loss/∂w  and  ∂loss/∂b
under some fixed values of x and y.
!!! To compute those derivatives, we call  loss.backward(), !!!
and then retrieve the values from w.grad and b.grad:
"""
loss.backward()
print(w.grad)
print(b.grad)

# tensor([[0.1230, 0.0802, 0.3322],
#         [0.1230, 0.0802, 0.3322],
#         [0.1230, 0.0802, 0.3322],
#         [0.1230, 0.0802, 0.3322],
#         [0.1230, 0.0802, 0.3322]])
# tensor([0.1230, 0.0802, 0.3322])

"""
We can only obtain the grad properties for the leaf nodes of the computational graph,
which have requires_grad property set to True.
For all other nodes in our graph, gradients will not be available.

We can only perform gradient calculations using backward once on a given graph,
for performance reasons.
If we need to do several backward calls on the same graph,
we need to pass  `retain_graph=True`  to the  .backward()  call.
"""
#%% Disabling Gradient Tracking
"""
By default, all tensors with  `requires_grad=True`
are tracking their computational history
and support gradient computation.

However, there are some cases when we do not need to do that,
for example, when we have trained the model and just want to apply it to some input data,
i.e. we only want to do  forward computations  through the network.

We can stop tracking computations by surrounding our computation code
with torch.no_grad() block:
"""
z = torch.matmul(x, w) + b
z.requires_grad         # True

with torch.no_grad():
    z = torch.matmul(x, w)+b
z
z.requires_grad         # False

"""
There are reasons you might want to disable gradient tracking:
1. To mark some parameters in your neural network as  __frozen parameters__.
   This is a very common scenario for  __finetuning a pretrained network__
2. To speed up computations when you are only doing   forward pass,
   because computations on tensors that do not track gradients would be more efficient.
"""
#%% More on Computational Graphs
"""
Conceptually,  autograd  keeps a record of data (tensors)
and all executed operations (along with the resulting new tensors)
in a directed acyclic graph (DAG) consisting of Function objects.
In this DAG,  leaves  are the input tensors,  roots  are the output tensors.
By tracing this graph from roots to leaves,
you can automatically compute the gradients using the chain rule.

In a  forward pass,  autograd  does two things simultaneously:

    - run the requested operation to compute a resulting tensor
    - maintain the operation’s gradient function in the DAG.

The backward pass kicks off when .backward() is called  __on the DAG root__.
autograd  then:

    - computes the gradients from each .grad_fn,
    - accumulates them in the respective tensor’s .grad attribute
    - using the chain rule, propagates all the way to the leaf tensors.

Note
DAGs are dynamic in PyTorch
An important thing to note is that the graph is recreated from scratch;
after each .backward() call,  autograd  starts populating a new graph.
This is exactly what allows you to use control flow statements in your model;
!!!  you can change the shape, size and operations at every iteration if needed.  !!!
"""


#%% Tensor Gradients and Jacobian Products
"""
In many cases, we have a scalar loss function,
and we need to compute the gradient with respect to some parameters.

However, there are cases when the output function is an arbitrary tensor.
In this case, PyTorch allows you to compute so-called  Jacobian product,
and not the  actual gradient.

For a vector function y⃗=f(x⃗), where x⃗=⟨x1,…,xn⟩ and y⃗=⟨y1,…,ym⟩
a gradient of y⃗​ with respect to x⃗ is given by  Jacobian matrix:

J=(∂y1/∂x1 ⋯ ∂y1/∂xn
       ⋮   ⋱   ⋮
   ∂ym/∂x1 ⋯ ∂ym/∂xn)

Instead of computing the  Jacobian matrix itself,
PyTorch allows you to compute  Jacobian Product
v^T⋅J  for a given input vector  v=(v1,…,vm).
This is achieved by calling  backward()  with `v` as an argument.
!!!  The size of `v` should be the same as the size of the original tensor,  !!!
with respect to which we want to compute the product:
"""
inp = torch.eye(5, requires_grad=True)
inp
out = (inp + 1).pow(2)
out
out.grad_fn     # <PowBackward0 at 0x7f763020e700>
out.grad        # None   .backward() was not called yet

out.backward( torch.ones_like(inp), retain_graph=True)
print(f"First call\n{inp.grad}")

out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")

inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")

#%%
"""
Notice that when we call backward for the second time with the same argument,
the value of the gradient is different.
This happens because when doing backward propagation,
PyTorch accumulates the gradients,
i.e. the value of computed gradients is added to the grad property
of all leaf nodes of computational graph.
If you want to compute the proper gradients,
you need to zero out the grad property before.
In real-life training an optimizer helps us to do this.

Note
Previously we were calling backward() function without parameters.
!!!  This is essentially equivalent to calling   backward(torch.tensor(1.0)),   !!!
which is a useful way to compute the gradients in case of a scalar-valued function,
such as loss during neural network training.
"""
