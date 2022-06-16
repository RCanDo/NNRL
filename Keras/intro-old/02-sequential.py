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
%reset

#%%
import numpy as np

import keras
import keras.backend as K

from keras.models import Sequential
from keras.layers import Dense, Activation

#%%
"""
The Sequential model is a linear stack of layers.
You can create a Sequential model by passing a list of layer instances to the constructor:
"""

model = Sequential([
            Dense(32, input_shape=(784,)),
            Activation('relu'),
            Dense(10),
            Activation('softmax')
        ])
model
model.summary()

#%%
model.evaluate()    #! RuntimeError: You must compile your model before training/testing. Use `model.compile(optimizer, loss)`.
                    #old: ValueError: If evaluating from data tensors, you should specify the `steps` argument.

K.eval(model)       # AttributeError: 'Sequential' object has no attribute 'eval'

#%% You can also simply add layers via the .add() method:

model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))

model.summary()

#%%  the following snippets are strictly equivalent:

model = Sequential()
model.add(Dense(32, input_dim=784))

model = Sequential()
model.add(Dense(32, input_shape=(784,)))

model.weights
model.weights[1]
K.eval(model.weights[1])

#%% Compilation

# for multi-class classification problem
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# for a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# for a mean squared error regression problem
model.compile(optimizer='rmsprop',
              loss='mse')

#%%
# for custom metrics
"""
metrics must take two arguments
    y_true: True labels. Theano/TensorFlow tensor.
    y_pred: Predictions. Theano/TensorFlow tensor of the same shape as y_true.
"""
def mean_pred(y_true, y_pred):
    return K.mean(abs(y_true - y_pred))

y_true = np.array([1, 2, 3])
y_pred = np.array([.2, 1.3, 3.1])

mean_pred(y_true, y_pred)   #! AttributeError: 'numpy.dtype' object has no attribute 'base_dtype'
K.mean(y_pred)              #! AttributeError: 'numpy.dtype' object has no attribute 'base_dtype'

"""
so y_true and y_pred must be turned to tf.tensor
"""
from tensorflow import constant

constant(y_true) - constant(y_pred)
    #! InvalidArgumentError: cannot compute Sub as input #1(zero-based) was expected to be a int32 tensor but is a double tensor [Op:Sub]
    #! ???
constant(y_true - y_pred)

K.mean(constant(y_pred))

def mean_pred(y_true, y_pred):
    return K.mean(constant(abs(y_true - y_pred)))  #!!! NOTICE the order of functions

K.eval(K.mean(constant(y_pred)))
K.eval(mean_pred(y_true, y_pred))

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy', mean_pred])

#%% Training

# binary classification

D = 100  # number of explanatory variables - size/dim of input
N = 1000 # number of records/observations - size of input
L = 2    # number of different labels i.e. values of explained varieble Y

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=D))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# dummy data
data = np.random.random((N, D))
labels = np.random.randint(L, size=(N, 1))

# train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, epochs=10, batch_size=32)

# model.train(data, labels, epochs=10, batch_size=32)
  # ! AttributeError: 'Sequential' object has no attribute 'train'

pred = model.predict(data)
pred
type(pred)
pred.shape

model.weights
model.weights[1]
K.eval(model.weights[1])

model.weights[0]
K.eval(model.weights[0]).shape

model.output           # <tf.Tensor 'dense_11/Sigmoid:0' shape=(None, 1) dtype=float32>
K.eval(model.output)   #! AttributeError: 'Tensor' object has no attribute 'numpy'

# ???


#%%
# categorical classification - 10 classes

L = 10

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=D))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# dummy data
data = np.random.random((N, D))
labels = np.random.randint(L, size=(N, 1))

# convert labels to categorical one-hot encoding
one_hot_labels = keras.utils.to_categorical(labels, num_classes=L)
one_hot_labels[range(5), :]

# train the model...
model.fit(data, one_hot_labels, epochs=10, batch_size=32)

predict = model.predict(data)
predict.shape
predict[range(5), :]
predict[range(5), :].argmax(axis=1)

#%%
mse = keras.losses.mean_squared_error(labels, predict)
mse.shape
mse
K.eval(mse)  # ???

mmse = keras.metrics.mean_squared_error(labels, predict)
K.eval(mmse)

#? MSE for categorical data ??? How it is calculated?

#%%
import tensorflow.keras as tfk

dir(tfk.losses)

tfmse = tfk.losses.MeanSquaredError()  #!!! MODERN version TF2
dir(tfmse)
res  = tfmse(labels, predict)    # <tf.Tensor: shape=(), dtype=float32, numpy=28.313793>
dir(res)
res.numpy()  # 28.313793

tfmse = tfk.losses.mean_squared_error(labels, predict)
tfmse  # <tf.Tensor: shape=(1000,), dtype=float32, numpy= array([...], dtype=float32)>

tfmse = tfk.losses.mse(labels, predict)
tfmse  # "

tfmmse = tfk.metrics.mse(labels, predict)
tfmmse # "


#%%
#%% EXAMPLES

#%% 1. Multilayer Perceptron (MLP) for multi-class softmax classification