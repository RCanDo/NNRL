# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title:
subtitle:
version: 1.0
type: tutorial
keywords: [tensor, Keras, Tensor Flow]
description: |
remarks:
    - OUTDATED; see `Intro For Engeneers`
sources:
    - link: https://keras.io
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    name:
    path: E:/ROBOCZY/Python/Keras/Intro-old
    date: 2021-09-13
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
"""

#%%
import numpy as np

from keras import backend as K
import tensorflow as tf

#%%
K.backend()

#%%
"""
The code below instantiates an input `placeholder`.
It's equivalent to tf.placeholder() or tf.tensor.matrix(), tf.tensor.tensor3(), etc.
"""
inputs = K.placeholder(shape=(2, 4, 5))
inputs

inputs = K.placeholder(shape=(None, 4, 5))
inputs

inputs = K.placeholder(ndim=3)
inputs

#%%
val = np.random.random((2, 3, 4))
val

var = K.variable(value=val)
var

#%% In Keras as in TensorFlow to see the results you must first evaluate it:
# (NOT TRUE ANYMORE, since TF2)

var # just some meta info about object
    # not now -- whole variable visible

# to see its value you must evaluate it
K.eval(var)   # still works!

#%%  via TensorFlow it's more complicated as one must create a `session` first
# (NOT TRUE ANYMORE, since TF2)

ss = tf.Session()  #! AttributeError: module 'tensorflow' has no attribute 'Session'
ss

init = tf.global_variables_initializer()  # and variables need to be initialized
ss.run(init)
print(ss.run(var))

#%%
var = K.zeros(shape=(3, 4, 5))
var
K.eval(var)   #! not necessary now

# via TF
ss.run(tf.global_variables_initializer())   #! OLD
ss.run(var)                                 #! OLD

#%%
var = K.ones(shape=(3, 4, 5))
var
K.eval(var)

#%%
K.clear_session()

# Initializing Tensors with Random Numbers
b = K.random_uniform_variable(shape=(3, 4), low=0, high=1) # Uniform distribution
c = K.random_normal_variable(shape=(3, 4), mean=0, scale=1) # Gaussian distribution
d = K.random_normal_variable(shape=(3, 4), mean=0, scale=1)

for l in (b, c, d): print(K.eval(l), end="\n\n")

list(map(lambda x: K.eval(x), [b, c, d]))


#%%
# Tensor Arithmetic

a = b + c * K.abs(d)            # elementwise
f = K.dot(a, K.transpose(b))    # matrix multiplication
g = K.sum(b, axis=1)            #
h = K.softmax(b)
k = K.concatenate([b, c], axis=-1)

for l in (a, f, g, h, k): print(K.eval(l), end="\n\n")

#%% ...
K.clear_session()

var = K.variable([[1, 2, 3], [4, 5, 6]])
var
K.eval(var)

var_transposed = K.transpose(var)
K.eval(var_transposed)

inputs = K.placeholder((2, 3))
inputs
input_transposed = K.transpose(inputs)
input_transposed

K.eval(input_transposed)  # ooops...

#%%
K.epsilon()

#%%
