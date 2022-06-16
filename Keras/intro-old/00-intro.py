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
import keras
from keras.models import Sequential

#%%
model = Sequential()
model

#%%
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model
model.summary()

#%% .compile()
"""
!!!
Once your model looks good, configure its learning process with .compile():
"""
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
model

#%% more refined
model.compile(loss      = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
model
model.summary()

#%% You can now iterate on your training data in batches:

# x_train and y_train are Numpy arrays -- just like in the Scikit-Learn API.

model.fit(x_train, y_train, epochs=5, batch_size=32)


#%% Alternatively, you can feed batches to your model manually:

model.train_on_batch(x_batch, y_batch)


#%% Evaluate your performance in one line:

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)


#%% Or generate predictions on new data:

classes = model.predict(x_test, batch_size=128)

#%%