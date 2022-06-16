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
sources:
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    name:
    path: E:/ROBOCZY/Python/Keras/Intro-old/
    date: 2019-04-04
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
"""
#%%
import numpy as np
from pprint import pprint

import keras
from keras import backend as K
from keras.layers import Dense
from keras.models import Sequential

K.backend()

#%%
def tprint(table, form="{:10.4f}", sep=["", "  |"]):
    for row in table:
        sep = np.resize(sep, len(row))
        for col, s in zip(row, sep):
            print(form.format(col), end=s)
        print()

#%%
"""
we'd like to implement the table below as a neural network (NN)
"""

table = np.random.randn(10, 2) * 10
table
tprint(table)

"""
Notice that every column of this table may be considered as a function of x \in [0, ..., 9], i.e.
c_k:[0, ..., 9] -> R  where  k \in [0, 1]
(M columns in table).
If we have N x M table then we need to construct NN
NN:[0, ..., N-1] -> R^m
i.e. it should have one input and M outputs.

Below we show that for a NN with one input and M outputs
it is enough to have one hidden layer with N neurons
to implement N x M table.

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

ww = np.linalg.solve(mmrelu, table)

np.dot(mmrelu, ww)
table               # OK !!!

#%%
"""
model not to train -- weights calculated algebraically
reference model -- the best possible -- perfect predictions
"""

model0 = Sequential()
model0.add(Dense(10, activation="relu", input_dim=1))
model0.add(Dense(2, activation='linear'))

model0
model0.summary()


#%% by the way:  configuration -- many things to learn!
model0.get_config()
pprint(model0.get_config())
print(model0.to_json())
print(model0.to_yaml())

#%%
model0.get_weights()

weights = [np.ones((1, 10)),     np.array(biases), ww, np.zeros(2)]
        # [np.array([[1] * 10]), biases,           ww, np.array([0, 0])]
model0.set_weights(weights)


"""
OK !!!   perfect solution -- this network is infallible :)

This is only example of possible perfect solution
--- there are infinitely many perfect solutions
--- it's just simple algebra!


But it wasn't learned -- it was just said what to do -- weights were calculated!

The problem is if the network with one layer is able to learn a table?
"""

tprint(np.concatenate((model0.predict(x), table), axis=1))


#%%
"""
Now we shall try to create models which are able to learn the table by its own.
Questions:
    - Is there a universal set of parameters which makes a NN capable of learning ANY table
      in a reasonable time?
    - ... which protects a learning alg. against falling into a cycle?
"""

#%%
#%% model 01

# construction
model01 = Sequential()
model01.add(Dense(10, activation="relu", input_dim=1))
model01.add(Dense(2, activation='linear'))
model01.compile( optimizer=keras.optimizers.SGD(nesterov=True, lr=.01, momentum=0.5, decay=0.)
               , loss='mse'
               , metrics=['mse'])

#%% learning
"""
Run it few times till learning will effectively stop, i.e.
mean_square_error stabilises or fall into cycles
what means that optimizing algorithm got into cycle or into a local minimum.
"""
model01.fit(x, table, epochs=int(1e4))

#%% GOF -- Goodness Of Fit:

table_ = model01.predict(x)
diff = table - table_
ratio = table_ / table
print("       prediction               table                difference               ratio")
tprint(np.concatenate((table_, table, diff, ratio), 1))

# conclusion: bad!

#%%
"""
Try to initialize the NN many times and observe how learning speed is changing.
WHY???
Because weights are initialised at random thus every new NN begins from a different point.
"""

#%%
#%% model 02

# construction
model02 = Sequential([Dense(10, activation="relu", input_dim=1),
                      Dense(2, activation='linear')])
model02.compile( optimizer=keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.)
               , loss='mse'
               , metrics=['mse'])

#%% learning
model02.fit(x, table, epochs=int(1e4))

#%% GOF
diff = table - model02.predict(x)
ratio = model02.predict(x) / table
print("       prediction               table                difference               ratio")
tprint(np.concatenate((model02.predict(x), table, diff, ratio), 1))

# conclusion: little better ?

#%%
#%% model 03

# construction
model03 = Sequential([Dense(10, activation="relu", input_dim=1),
                      Dense(2, activation='linear')])
model03.compile( optimizer=keras.optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.01)
               , loss='mse'
               , metrics=['mse'])

#%% learning
model03.fit(x, table, epochs=int(1e4))

#%% GOF
diff = table-model03.predict(x)
ratio = model03.predict(x)/table
print("       prediction               table                difference               ratio")
tprint(np.concatenate((model03.predict(x), table, diff, ratio), 1))

# conclusion: better

#%%
#%% model 04

# construction
model04 = Sequential([Dense(10, activation="relu", input_dim=1,
                            bias_initializer='zeros',
                            kernel_initializer='ones'),
                      Dense(2, activation='linear',
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros')])
model04.compile( optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
               , loss='mse'
               , metrics=['mse'])

#%% learning
model04.fit(x, table, epochs=int(1e4))

#%% GOF
diff = table-model04.predict(x)
ratio = model04.predict(x)/table
print("       prediction               table                difference               ratio")
tprint(np.concatenate((model04.predict(x), table, diff, ratio), 1))

"""conclusion: better
"""
#%%


#%% model 05

lay1 = Dense(10, activation='relu', input_dim=1)
dir(lay1)
lay1.trainable = False

predictions = Dense(2, activation='linear')(lay1)

model3 = Model(inputs=inputs, outputs=predictions)

model3.compile( optimizer=keras.optimizers.SGD(lr=0.1, momentum=0.1, decay=0.0, nesterov=True)
              , loss='mse'
              , metrics=['mse']
              )

model3.summary()

#%%
# construction
model05 = Sequential()
model05.add(Dense(10, activation="relu", input_dim=1))
model05.layers[0].set_weights([np.ones((1, 10)), np.array(bias)])

model05.add(Dense(2, activation='linear'))

model05.weights

model05.layers[0].trainable = False
model05.weights

#!!! notice how the order of layers has changed !!!

K.eval(model05.weights[2])
K.eval(model05.weights[3])

model05.compile( optimizer=keras.optimizers.SGD(nesterov=True, lr=.02, momentum=0.5, decay=0.)
               , loss='mse'
               , metrics=['mse'])


#%% learning
model05.fit(x, table, epochs=1000)

#%% gof
diff = table-model05.predict(x)
diff_ratio = diff/table
print("       prediction               table                difference            diff_ratio")
tprint(np.concatenate((model05.predict(x), table, diff, diff_ratio), 1))

"""conclusion: bad!
"""
#%%

model.predict(x)
table

K.eval(model.weights[0])
K.eval(model.weights[1])
K.eval(model.weights[2])
K.eval(model.weights[3])

model.get_config()

#%%

model.layers[i].set_weights(listOfNumpyArrays)
model.get_layer(layerName).set_weights(...)
model.set_weights(listOfNumpyArrays)



#%%
