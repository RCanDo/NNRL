#! python3
"""
---  
title: Convolutional Neural Networks
subtitle: 
version: 1.0
type: example
keywords: [callback, TensorBorad, learning statistics]
description: |
    ...
remarks: 
todo:
sources: 
    - title: Deep Learning basics with Python, TensorFlow and Keras
      chapter: p.3 - Convolutional Neural Networks
      link: https://pythonprogramming.net/tensorboard-analysis-deep-learning-python-tensorflow-keras/
      usage: | 
          Copy of the code to see how it works.
file:
    usage:
        interactive: True
        terminal: False
    name: "03 - Convolutional Neural Networks.py"
    path: ./Python/TensorFlow/Intro To Deep Learning/
    date: 2019-06-20 
    authors: 
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: arkadiusz.kasprzyk@tieto.com
    project:         
    company: Tieto
"""  

#%%

cd "D:/ROBOCZY/Python/TensorFlow/Intro To Deep Learning/"
ls

#%%

import tensorflow as tf
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import pickle

#%%

from load_cats_and_dogs import *

# we shall use all data (quite a lot!!!)
X, y, X_val, y_val = load_cats_and_dogs(-1, -1)

#%%

model = Sequential()

model.add(Conv2D(256, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X, y, 
          batch_size=32, 
          epochs=3, 
          #validation_split=0.3
          validation_data=(X_val, y_val)
          )
