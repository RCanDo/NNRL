#! python3
"""
---  
title: Recurrent Neural Networks
subtitle: 
version: 1.0
type: example
keywords: [callback, TensorBorad, learning statistics]
description: |
remarks: 
todo:
sources: 
    - title: Deep Learning basics with Python, TensorFlow and Keras
      chapter: p.7 - Recurrent Neural Networks
      link: https://pythonprogramming.net/tensorboard-analysis-deep-learning-python-tensorflow-keras/
      usage: | 
          Copy of the code to see how it works.
file:
    usage:
        interactive: True
        terminal: False
    name: "07 - Recurrent Neural Networks.py"
    path: ./Python/TensorFlow/Intro To Deep Learning/
    date: 2019-06-26 
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
"""
The idea of a Recurrent Neural Network is that sequences and order matters.
"""

import tensorflow as tf
from tensorflow.keras.datasets import mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM #, CuDNNLSTM

#%%

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test

x_train = x_train/255.0
x_test = x_test/255.0

print(x_train.shape)
print(x_train[0].shape)

#%%

model = Sequential()

# IF you are running with a GPU, try out the CuDNNLSTM layer type instead (don't pass an activation, tanh is required)
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

#%%

model.fit(x_train[:1e3],
          y_train[:1e3],
          epochs=3,
          validation_data=(x_test, y_test))
