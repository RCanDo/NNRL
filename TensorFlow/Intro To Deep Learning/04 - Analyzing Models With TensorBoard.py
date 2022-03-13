#! python3
"""
---  
title: Analyzing Models with TensorBoard
subtitle: Example of usage of keras.callback.TensorBoard()
version: 1.0
type: example
keywords: [callback, TensorBorad, learning statistics]
description: |
    How to use keras.callback.TensorBoard() in order to have good monitoring of the learnig process.
remarks: 
todo:
sources: 
    - title: Deep Learning basics with Python, TensorFlow and Keras
      chapter: p.4 - Analyzing Models with TensorBoard
      link: https://pythonprogramming.net/tensorboard-analysis-deep-learning-python-tensorflow-keras/
      usage: | 
          Copy of the code to see how it works.
file:
    usage:
        interactive: True
        terminal: False
    name: "04 - Analyzing Models With TensorBoard.py"
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
              metrics=['accuracy'],
              )

#%%

model.fit(X, y,
          batch_size=32,
          epochs=10,
          #validation_split=0.3,
          validation_data=(X_val, y_val),
          callbacks=[tensorboard])

#%%
"""
Now we can see how our model did over time. Let's change some things in our model. 
To begin, we never added an activation to our dense layer. 
Also, let's try a smaller model overall:
"""

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape=X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

#%%
"""
Among other things, we also change the name to 
"""
NAME = "Cats-vs-dogs-64x2-CNN"
for file in glob.glob("./logs/Cats-vs-dogs-64x2-CNN/events*"): os.remove(file)
"""
Don't forget to do this, or you'll append to your previous model's logs 
instead by accident and it wont look too good.
"""
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'],
              )

model.fit(X, y,
          batch_size=32,
          epochs=10,
          #validation_split=0.3,
          validation_data=(X_val, y_val),
          callbacks=[tensorboard])

#%%



