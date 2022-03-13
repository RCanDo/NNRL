# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 22:24:13 2019

@author: kasprark
"""

%reset


#%%

import numpy as np
from keras import backend as K
from keras import utils
import tensorflow as tf

from keras.models import Sequential, Model
from keras.layers import Dense, Activation

from keras.initializers import RandomNormal
from keras import layers, optimizers

#%%

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


#%%

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

# dummy data
data = np.random.random((1000, 100))
labels = np.random.randint(10, size=(1000, 1))

# convert labels to categorical one-hot encoding
one_hot_labels = utils.to_categorical(labels, num_classes=10)

# train the model...
model.fit(data, one_hot_labels, epochs=10, batch_size=32)

predict = model.predict(data)
predict.shape


#%%  
#%%
#%%

# !!! RESTART KERNEL FIRST !!!

cd "D:\ROBOCZY\Python\TensorFlow\"
pwd
ls

#%%
#%%  Setup and basic usage
"""
To start eager execution, add tf.enable_eager_execution() 
to the beginning of the program or console session. 
"""
# !!! Do not add this operation to other modules that the program calls `couse you'll get error:
#  ValueError: tf.enable_eager_execution must be called at program startup.

from __future__ import absolute_import, division, print_function

import tensorflow as tf
from tensorflow.keras import layers

import numpy as np

tf.enable_eager_execution()   # !!! RESTART KERNEL FIRST !!!


#%% Now you can run TensorFlow operations and the results will return immediately: 

tf.executing_eagerly() 


#%%

model = tf.keras.Sequential([
        # Adds a densely-connected layer with 64 units to the model:
        layers.Dense(10, activation='relu', input_shape=(5,)),
        # Add another:
        layers.Dense(3, activation='softmax')
        ])

np_data = np.random.random((10, 5))
model(np_data)  # InvalidArgumentError: cannot compute MatMul ...
model.predict(np_data)  # OK
model(np_data[0])  # InvalidArgumentError: cannot compute MatMul ...
model.predict(np_data[0])  # ValueError: Error when checking input:
model.predict(np_data[:2])  # OK

model.predict(np_data)  # OK
model.weights       #! weights do not change !!!


tf_data = tf.random_uniform((10, 5))
model(tf_data)  # OK
model.predict(tf_data)  # InvalidArgumentError: Index out of range using input dim 2
model(tf_data[0])  # InvalidArgumentError: In[0] is not a matrix.
model.predict(tf_data[0])  # ValueError: Error when checking input:
model.predict(tf_data[:2])  # InvalidArgumentError: In[0] is not a matrix

model(tf_data)  # OK
model.weights       #! weights do not change !!!


#%%

np_data_slices = tf.data.Dataset.from_tensor_slices(np_data)
np_data_slices = dataset.batch(32)
np_data_slices = dataset.repeat()

model(np_data_slices) # ERROR ...




np_labels = np.random.random((10, 3))


# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((np_data, np_labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()



# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)



