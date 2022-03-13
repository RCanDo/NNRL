#! python3
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Quickstart For Beginners
part: 1
subtitle:
version: 1.0
type: code chunks
keywords: [TensorFlow, quickstart, beginners, MNIST]
description: Examples.
sources:
    - title: TensorFlow 2 quickstart for beginners
      link: https://www.tensorflow.org/tutorials/quickstart/beginner/
      usage: copy
file:
    usage:
        interactive: True
        terminal: False
    name: "01 - Beginners.py"
    path: "D:/ROBOCZY/Python/TensorFlow2/Tutorials/Quickstart/"
    date: 2019-10-02
    authors:
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email:
              - arkadiusz.kasprzyk@tieto.com
              - akasp666@google.com
              - arek@staart.pl
"""


#%%
"""
This short introduction uses Keras to:
    Build a neural network that classifies images.
    Train this neural network.
    And, finally, evaluate the accuracy of the model.

This is a Google Colaboratory notebook file.
Python programs are run directly in the browser—a great way to learn and use TensorFlow.
To follow this tutorial, run the notebook in Google Colab by clicking the button
at the top of this page.
    In Colab, connect to a Python runtime: At the top-right of the menu bar, select CONNECT.
    Run all the notebook code cells: Select Runtime > Run all.

Download and install the TensorFlow 2 package. Import TensorFlow into your program:
"""
#%%
# from __future__ import absolute_import, division, print_function, unicode_literals

# Install TensorFlow
import tensorflow as tf


#%% Load and prepare the MNIST dataset.
# Convert the samples from integers to floating-point numbers:

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

type(x_train)
x_train.shape
x_train[1, :, :]

y_train.shape
y_train[:7]

#%% Build the tf.keras.Sequential model by stacking layers.
# Choose an optimizer and loss function for training:

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#%% Train and evaluate the model:

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)
# 10000/1 - 2s - loss: 0.0398 - accuracy: 0.9773
# [0.06980192945897579, 0.9773]
"""
The image classifier is now trained to ~98% accuracy on this dataset.
To learn more, read the TensorFlow tutorials.
"""

model(x_train[:4])
model.predict(x_train[:4])    # "

#%%
"""
If you want your model to return a probability,
you can wrap the trained model, and attach the softmax to it:
"""
probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probability_model(x_test[:5])
probability_model(x_test[:5]).numpy().argmax(axis=1)
y_test[:5]

#%%