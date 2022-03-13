#! python3
"""
---  
title: How to use your trained model
subtitle: 
version: 1.0
type: example
keywords: [callback, TensorBorad, learning statistics]
description: |
    Harnessing a trained model to work.
remarks: 
todo:
sources: 
    - title: Deep Learning basics with Python, TensorFlow and Keras
      chapter: p.6 - How to use your trained model
      link: https://pythonprogramming.net/tensorboard-analysis-deep-learning-python-tensorflow-keras/
      usage: | 
          Copy of the code to see how it works.
file:
    usage:
        interactive: True
        terminal: False
    name: "06 - How To Use Your Trained Model.py"
    path: ./Python/TensorFlow/Intro To Deep Learning/
    date: 2019-06-25 
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

import numpy as np
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
import os, glob
os.chdir("D:/ROBOCZY/Python/TensorFlow/Intro To Deep Learning/")

"""{anaconda-prompt}
tensorboard --logdir ./logs2 --host 127.0.0.1
"""

#%%

from load_cats_and_dogs import *

X, y, X_val, y_val = load_cats_and_dogs(2054, 512)


#%%

"""After investigating TB (lesson 05) we've found that the best models are:
"""
models = {"3-conv-32-nodes-0-dense" : (3, 32, 0),     # best
          "3-conv-64-nodes-0-dense" : (3, 64, 0),
          "3-conv-32-nodes-1-dense" : (3, 32, 1),
          "2-conv-32-nodes-2-dense" : (2, 32, 2)      # weakest
         }

#%%

for mod_name in models.keys():

    NAME = "{}-{}".format(mod_name, int(time.time()))
    print(NAME)
    #tensorboard = TensorBoard(log_dir="logs2/{}".format(NAME))  # first run 20 epochs - too much!
                                                                 # models overfitted
    #tensorboard = TensorBoard(log_dir="logs3/{}".format(NAME)) 
    tensorboard = TensorBoard(log_dir="logs4/{}".format(NAME)) 

    conv_layer, layer_size, dense_layer = models[mod_name]

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
    
    model.save("{}-1.model".format(mod_name))

#%%
"""
From TB it shows up that the last model "2-conv-32-nodes-2-dense" is very bad.
It also confirms that the first "3-conv-32-nodes-0-dense" is the best.
"""
    
#%%
    
#model = tf.keras.models.load_model("3-conv-32-nodes-0-dense.model")
model = tf.keras.models.load_model("3-conv-32-nodes-0-dense-1.model")
model.summary()

#%%

import cv2

def prepare(filepath):
    IMG_SIZE = 100  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

#%%
    
CATEGORIES = ["Dog", "Cat"]

prediction = model.predict([prepare('cat-800x600.jpg')])
print(prediction)  # will be a list in a list.
print(CATEGORIES[round(prediction[0][0])])

# ...ooops...

#%%

X, y, X_val, y_val = load_cats_and_dogs(1, 8)

y_hat = model.predict(X_val).reshape(8)

for y, y_ in zip(y_val, y_hat):
    print("{} - {:1.2f} - abs.error {:.2f}".format(y, y_, abs(y_ - y)))

#%%



