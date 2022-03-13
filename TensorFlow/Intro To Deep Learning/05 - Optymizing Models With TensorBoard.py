#! python3
"""
---  
title: Optymizing Models with TensorBoard
subtitle: Example of usage of keras.callback.TensorBoard()
version: 1.0
type: example
keywords: [callback, TensorBorad, learning statistics]
description: |
    How to use TensorBoard to optymize models.
remarks: 
todo:
sources: 
    - title: Deep Learning basics with Python, TensorFlow and Keras
      chapter: p.5 - Optymizing Models with TensorBoard
      link: https://pythonprogramming.net/tensorboard-analysis-deep-learning-python-tensorflow-keras/
      usage: | 
          Copy of the code to see how it works.
file:
    usage:
        interactive: True
        terminal: False
    name: "05 - Optymizing Models With TensorBoard.py"
    path: ./Python/TensorFlow/Intro To Deep Learning/
    date: 2019-06-20 
    authors: 
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: arkadiusz.kasprzyk@tieto.com
    project: AIML Competence Development / Gym Training             
    company: Tieto
"""  

#%%
"""{cmd}
cd "D:/ROBOCZY/Python/TensorFlow/Intro To Deep Learning/"
ls
"""
#%%

import tensorflow as tf
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
# more info on callbakcs: https://keras.io/callbacks/ model saver is cool too.
from tensorflow.keras.callbacks import TensorBoard
import pickle
import time

import random

#%%

dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)

#%%
import os, glob
os.chdir("D:/ROBOCZY/Python/TensorFlow/Intro To Deep Learning/")
for file in glob.glob("./logs/Cats-vs-dogs-CNN/events*"): os.remove(file)

""" recursively:
for folder, subfolders, files in os.walk("./logs"):
    for file in glob.glob(os.path.join(subfolder, "events*")): os.remove(file)
"""

NAME = "Cats-vs-dogs-CNN"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

"""{anaconda-prompt}
tensorboard --logdir .\logs --host 127.0.0.1
"""

#%%

from load_cats_and_dogs import *

X, y, X_val, y_val = load_cats_and_dogs(1024, 256)

#%%

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:

            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)
            tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

            model = Sequential()

            model.add(Conv2D(layer_size, (3, 3), input_shape=X.shape[1:]))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))

            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3, 3)))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())

            for _ in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            model.add(Dense(1))
            model.add(Activation('sigmoid'))


            model.compile(loss='binary_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'],
                          )

            model.fit(X, y,
                      batch_size=32,
                      epochs=10,
                      shuffle=True,
                      validation_data=(X_val, y_val),
                      callbacks=[tensorboard])
            
#%%
"""
After investigating TB we find that the best models are:
3-conv-32-nodes-0-dense-{}  best
3-conv-64-nodes-0-dense-{}
3-conv-32-nodes-1-dense-{}
2-conv-32-nodes-2-dense-{}  weakest
"""
#%%
