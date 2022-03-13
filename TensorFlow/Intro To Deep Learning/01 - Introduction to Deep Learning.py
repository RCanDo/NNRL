#! python3
"""
---  
title: Introduction To Deep Learning
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
      chapter: p.1 - Introduction To Deep Learning
      link: https://pythonprogramming.net/tensorboard-analysis-deep-learning-python-tensorflow-keras/
      usage: | 
          Copy of the code to see how it works.
file:
    usage:
        interactive: True
        terminal: False
    name: "01 - Introduction To Deep Learning.py"
    path: ./Python/TensorFlow/Intro To Deep Learning/
    date: 2019-06-18 
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

import tensorflow.keras as keras
import tensorflow as tf

print(tf.__version__)


#%%

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train[0]
y_train[0]

x_train[0].shape   # (28, 28)
x_train.shape      # (6e4, 28,28)
y_train.shape      # (6e4)


#%%

import matplotlib.pyplot as plt

plt.imshow(x_train[0])   # 
plt.imshow(x_train[0], cmap=plt.cm.binary)   # bw picture 
plt.show()

#%%

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test  = tf.keras.utils.normalize(x_test,  axis=1)

print(x_train[0])

plt.imshow(x_train[0])
plt.imshow(x_train[0],cmap=plt.cm.binary)
plt.show()

#%%

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())    #! it's better this way -- do not use np.reshape()
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#%%

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%

model.fit(x_train, y_train, epochs=3)

#%%

val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)

#%%

model.save('epic_num_reader.model')  
#! NotImplementedError: Currently `save` requires model to be a graph network. 
#  Consider using `save_weights`, in order to save the weights of the model.

# Load it back:
new_model = tf.keras.models.load_model('epic_num_reader.model')

#%% 
# make predictions!

predictions = model.predict(x_test)

print(predictions)

import numpy as np

print(np.argmax(predictions[0]))

plt.imshow(x_test[0],cmap=plt.cm.binary)
plt.show()



#%%
"""
As of Dec 21st 2018, there's a known issue with the code
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()   # 28x28 numbers of 0-9
x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten())   #Flatten the images! Could be done with numpy reshape
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, input_shape= x_train.shape[1:]))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.summary()


#%%
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%
model.fit(x_train, y_train, epochs=3)

"""
ALL RIGHT! using  numpy.reshape()  is OK!
"""

#%%



