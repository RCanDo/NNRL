# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: NN table implementation
subtitle:
version: 1.0
type: tutorial
keywords: [tensor, Keras, Tensor Flow]
description: |
    Below we show that for a NN with one input and M outputs
    it is enough to have one hidden layer with N neurons
    to implement N x M table.
    But is such a small NN able to LEARN a table?
remarks:
    - version of 01-..
sources:
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    name:
    path: E:/ROBOCZY/Python/Keras/Intro-old/
    date: 2019-04-08
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
"""

import numpy as np

import keras
from keras import backend as K
from keras.layers import Dense, Activation
from keras.models import Model

K.backend()

#%%
"""
we'd like to implement the table below as a neural network (NN)
"""

table = np.random.randn(10, 2) * 10
table

"""
Notice that every column of this table may be considered as a function of x \in [0, ..., 9], i.e.
c_k:[0, ..., 9] -> R  where  k \in [0, 1]
(m columns in table).
If we have n x m table then we need to construct NN
N:[0, ..., n-1] -> R^m
i.e. it should have one input and m outputs.

Below we show that for a NN with one input and m outputs
it is eonugh to have one hidden layer with n neurons
to implement n x m table.

But is such a small NN able to LEARN a table?
"""

#%%

def relu(x):
    return K.eval(keras.activations.relu(x))

x = np.array(range(10))
x

relu(x)
relu(-x)

biases = np.array([-k for k in range(-1,9)])

mm = [[n + b for b in biases] for n in x ]

mmrelu = [relu(row) for row in mm]

np.linalg.solve(mmrelu, table[:, 0])

w2 = np.linalg.solve(mmrelu, table)

np.dot(mmrelu, w2)
table               # OK !!!

#%%
"""
model not to train -- weights calculated algebraically
"""

model = keras.Sequential()
model.add(Dense(10, activation='relu', input_dim=1))
model.add(Dense(2, activation='linear'))

model.summary()

#%% configuration -- many things to learn!
model.get_config()
print(model.to_json())
print(model.to_yaml())

#%%

model.get_weights()  # weights are generated at random

weights = [ np.array([[1] * 10]), biases, w2, np.array([0, 0]) ]

model.set_weights(weights)

model.predict(x)
table

## OK !!!   perfect solution -- this network is infallible :)
"""
This is only example of possible perfect solution
--- there are infinitely many perfect solutions
--- it's just simple algebra!

The problem is if the network with one layer is able to learn a table?
Below number of trials...
"""
#%%
#%%

model2 = keras.Sequential([Dense(10, input_dim=1),
                           Activation('relu'),
                           Dense(2),
                           Activation('linear')
                          ])

model2.summary()

model2.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

model2.get_weights()

#%%
#%%
def mse(y, yhat):
    return sum(sum((y - yhat) ** 2))/y.size

def fit(model, min_mse=.1, max_iter=100):
    MSE = 1
    k = 0
    while (MSE > .1 and k < max_iter):
        k += 1
        print(k, end=" : ")
        model.fit(x, table, epochs=100, batch_size=10)
        MSE = mse(table, model.predict(x))
        print(MSE)
    print(model.predict(x))
    print(table)

#%% learning model2

fit(model2, 10, 100)

#%%
#%%

inputs = keras.Input(shape=(1,))

lay1 = Dense(10, activation='relu')(inputs)
predictions = Dense(2, activation='linear')(lay1)

model3 = Model(inputs=inputs, outputs=predictions)

model3.compile( optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.1, decay=0.0, nesterov=True)
              , loss='mse'
              , metrics=['mse']
              )

model3.summary()

#%%

mfit3 = model3.fit(x, table, epochs=100, batch_size=10)

model3.get_weights()
model

#%%
dir(mfit3)
mfit3.model
mfit3.params
mfit3.validation_data
mfit3.history

#%%
#%%

lay = keras.layers.Dense(2, input_dim=10)
lay.non_trainable_weights
lay

lay.add_weight()
lay.bias
lay.weights

#%%

import keras

class TableNN(keras.Model):

    def __init__(self, rows=10, columns=2, train_first=False):

        super(TableNN, self).__init__(name='table_nn')

        #self.inputs = keras.Input((1,))

        self.dense1 = keras.layers.Dense(units=rows,
                           activation='relu',
                           kernel_initializer='glorot_uniform',
                           bias_initializer='zeros',
                           input_dim=1,
                           #
                           kernel_regularizer=None,
                           bias_regularizer=None,
                           activity_regularizer=None,
                           kernel_constraint=None,
                           bias_constraint=None)

        if not train_first:
            self.dense1.set_weights=[ np.ones(10), np.arange(1, -9, -1) ]
            self.dense1.trainable = False

        self.dense2 = keras.layers.Dense(units=columns,
                           activation='linear',
                           kernel_initializer='glorot_uniform',
                           bias_initializer='zeros',
                           kernel_regularizer=None,
                           bias_regularizer=None,
                           activity_regularizer=None,
                           kernel_constraint=None,
                           bias_constraint=None)

    def call(self, inputs):
        x = self.dense1(inputs)
        return self.dense2(x)

#%%

model = TableNN()

model.compile( optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.1, decay=0.0, nesterov=True)
             , loss='mse'
             , metrics=['mse']
             )
model.layers

model.summary()    #! ValueError: This model has not yet been built
model.fit(x, table, epochs=100)   #! AttributeError: 'TableNN' object has no attribute '_feed_output_names'





