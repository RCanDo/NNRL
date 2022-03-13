# -*- coding: utf-8 -*-
"""
file: 01-keras.py 
title: 
chapter: Keras
book: 
link: https://www.tensorflow.org/guide/keras
date: 2019-02-24 Sun 19:06:41
author: kasprark
email: arkadiusz.kasprzyk@tieto.com
"""

%reset

#%%

cd "D:\ROBOCZY\Python\TensorFlow\Guide\HighLevelAPI"
pwd
ls

# !pip install -q pyyaml  # Required to save models in YAML format

#%%

import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

"""
tf.keras can run any Keras-compatible code, but keep in mind:

- The tf.keras version in the latest TensorFlow release might not be the same 
  as the latest keras version from PyPI. Check tf.keras.version.
- When saving a model's weights, tf.keras defaults to the checkpoint format. 
  Pass save_format='h5' to use HDF5.
"""

#%% Build a simple model

#%% Sequential model
"""
In Keras, you assemble layers to build models. 
A model is (usually) a graph of layers. 
The most common type of model is a stack of layers: the tf.keras.Sequential model.
To build a simple, fully-connected network (i.e. multi-layer perceptron):
"""

model = tf.keras.Sequential()
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model
model.summary()  # ! ValueError: This model has never been called ...

#%%

# Create a sigmoid layer:
layers.Dense(64, activation='sigmoid')
# Or:
layers.Dense(64, activation=tf.sigmoid)

# A linear layer with L1 regularization of factor 0.01 applied to the kernel matrix:
layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))

# A linear layer with L2 regularization of factor 0.01 applied to the bias vector:
layers.Dense(64, bias_regularizer=tf.keras.regularizers.l2(0.01))

# A linear layer with a kernel initialized to a random orthogonal matrix:
layers.Dense(64, kernel_initializer='orthogonal')

# A linear layer with a bias vector initialized to 2.0s:
layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))


#%%
ld = layers.Dense(64, bias_initializer=tf.keras.initializers.constant(2.0))
ld
dir(ld)


#%% Train and evaluate

#%% Set up training

model = tf.keras.Sequential([
        # Adds a densely-connected layer with 64 units to the model:
        layers.Dense(64, activation='relu', input_shape=(32,)),
        # Add another:
        layers.Dense(64, activation='relu'),
        # Add a softmax layer with 10 output units:
        layers.Dense(10, activation='softmax')
        ])

model.summary()
    
model.compile(optimizer=tf.train.AdamOptimizer(0.001),  # tf.train.
              loss='categorical_crossentropy',          # tf.keras.losses.
              metrics=['accuracy']                      # tf.keras.metrics.
              )

# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae']   # mean absolute error
              )

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy]
              )

#%% Input NumPy data

import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.fit(data, labels, epochs=10, batch_size=32)

#%%  !!!

data = np.random.random((10, 32))
model(data)  # InvalidArgumentError: cannot compute MatMul ...
model.predict(data)  # OK
model(data[0])  # InvalidArgumentError: cannot compute MatMul ...
model.predict(data[0])  # ValueError: Error when checking input:
model.predict(data[:2])  # OK

data = tf.random_uniform((10, 32))
model(data)  # OK
model.predict(data)  # InvalidArgumentError: Index out of range using input dim 2
model(data[0])  # InvalidArgumentError: In[0] is not a matrix.
model.predict(data[0])  # ValueError: Error when checking input:
model.predict(data[:2])  # InvalidArgumentError: In[0] is not a matrix


data = tf.random_uniform((10, 32))
model(data)  # OK
model.weights

with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())
    print(ss.run(model.weights))


tf_data = tf.Variable(data)

with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())
    print(ss.run(model(tf_data)))

with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())
    print(ss.run(tf.arg_max(model(data)[0], 0)))


#%% ... with validation data

import numpy as np

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

val_data = np.random.random((100, 32))
val_labels = np.random.random((100, 10))

model.fit(data, labels, epochs=10, batch_size=32,
          validation_data=(val_data, val_labels))

#%% Input tf.data datasets
"""
Use the Datasets API to scale to large datasets or multi-device training. 
Pass a tf.data.Dataset instance to the fit method:
"""

# Instantiates a toy dataset instance:
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()

# Don't forget to specify `steps_per_epoch` when calling `fit` on a dataset.
model.fit(dataset, epochs=10, steps_per_epoch=30)
"""
Here, the fit method uses the `steps_per_epoch` argument
— this is the number of training steps the model runs before it moves to the next epoch.
Since the Dataset yields batches of data, this snippet does not require a batch_size.
"""

#%% Datasets can also be used for validation:

dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)


#%% Evaluate and predict
# To evaluate the inference-mode loss and metrics for the data provided:

data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))

model.evaluate(data, labels, batch_size=32)

model.evaluate(dataset, steps=30)

#%% And to predict the output of the last layer in inference for the data provided, 
# as a NumPy array:

result = model.predict(data)
result = model.predict(data, batch_size=32)
print(result.shape)


#%%
#%% Build advanced models

#%% Functional API
"""
The tf.keras.Sequential model is a simple stack of layers that cannot represent arbitrary models.
Use the Keras functional API to build complex model topologies such as:

    Multi-input models,
    Multi-output models,
    Models with shared layers (the same layer called several times),
    Models with non-sequential data flows (e.g. residual connections).

Building a model with the functional API works like this:

    A layer instance is callable and returns a tensor.
    Input tensors and output tensors are used to define a tf.keras.Model instance.
    This model is trained just like the Sequential model.

The following example uses the functional API to build a simple, fully-connected network:
"""

inputs = tf.keras.Input(shape=(32,))  # Returns a placeholder tensor

# A layer instance is callable on a tensor, and returns a tensor.
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(64, activation='relu')(x)
predictions = layers.Dense(10, activation='softmax')(x)

# Instantiate the model given inputs and outputs.

model = tf.keras.Model(inputs=inputs, outputs=predictions)
model.summary()

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs
model.fit(data, labels, batch_size=32, epochs=5)

#%% Model subclassing
"""
Build a fully-customizable model by subclassing tf.keras.Model 
and defining your own forward pass. 

    Create layers in the __init__() method and set them as attributes of the class instance. 
    Define the forward pass in the call() method.

Model subclassing is particularly useful when `eager` execution is enabled 
since the _forward pass_ can be written _imperatively_.

Key Point: Use the right API for the job. 
    While model subclassing offers flexibility, 
    it comes at a cost of greater complexity and more opportunities for user errors. 
    If possible, prefer the functional API.

The following example shows a subclassed tf.keras.Model using a custom forward pass:
"""

class MyModel(tf.keras.Model):

  def __init__(self, num_classes=10):
    super(MyModel, self).__init__(name='my_model')
    self.num_classes = num_classes
    # Define your layers here.
    self.dense_1 = layers.Dense(32, activation='relu')
    self.dense_2 = layers.Dense(num_classes, activation='sigmoid')

  def call(self, inputs):
    # Define your forward pass here,
    # using layers you previously defined (in `__init__`).
    x = self.dense_1(inputs)
    return self.dense_2(x)

  def compute_output_shape(self, input_shape):
    # You need to override this function if you want to use the subclassed model
    # as part of a functional-style model.
    # Otherwise, this method is optional.
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.num_classes                                # ???
    return tf.TensorShape(shape)

#%% ... Instantiate the new model class:
    
model = MyModel(num_classes=10)

model.summary()

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)

#%% Custom layers
"""
Create a custom layer by subclassing tf.keras.layers.Layer 
and implementing the following methods:

    build: Create the _weights_ of the layer. 
           Add weights with the add_weight() method.
    call: Define the forward pass.
    compute_output_shape: Specify how to compute the output shape of the layer 
        given the input shape.
        
Optionally, a layer can be _serialized_ by implementing the get_config() method 
and the from_config() class method.

Here's an example of a custom layer that implements a matmul of an input with a kernel matrix:
"""

class MyLayer(layers.Layer):

  def __init__(self, output_dim, **kwargs):
    self.output_dim = output_dim
    super(MyLayer, self).__init__(**kwargs)

  def build(self, input_shape):
    shape = tf.TensorShape((input_shape[1], self.output_dim))
    # Create a trainable weight variable for this layer.
    self.kernel = self.add_weight(name='kernel',
                                  shape=shape,
                                  initializer='uniform',
                                  trainable=True)
    # Make sure to call the `build` method at the end
    super(MyLayer, self).build(input_shape)

  def call(self, inputs):
    return tf.matmul(inputs, self.kernel)

  def compute_output_shape(self, input_shape):
    shape = tf.TensorShape(input_shape).as_list()
    shape[-1] = self.output_dim
    return tf.TensorShape(shape)

  def get_config(self):
    base_config = super(MyLayer, self).get_config()
    base_config['output_dim'] = self.output_dim
    return base_config

  @classmethod
  def from_config(cls, config):
    return cls(**config)


#%%  Create a model using your custom layer:

model = tf.keras.Sequential([MyLayer(10), layers.Activation('softmax')])

# The compile step specifies the training configuration
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Trains for 5 epochs.
model.fit(data, labels, batch_size=32, epochs=5)


#%%
#%%  Callbacks
"""
A callback is an object passed to a model to customize and extend its behavior during training. 
You can write your own custom callback, or use the built-in `tf.keras.callbacks` that include:

    tf.keras.callbacks.ModelCheckpoint: Save checkpoints of your model at regular intervals.
    tf.keras.callbacks.LearningRateScheduler: Dynamically change the learning rate.
    tf.keras.callbacks.EarlyStopping: Interrupt training when validation performance has stopped improving.
    tf.keras.callbacks.TensorBoard: Monitor the model's behavior using TensorBoard.

To use a tf.keras.callbacks.Callback, pass it to the model's fit method:
"""

callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  tf.keras.callbacks.TensorBoard(log_dir='./logs')
]

model.fit(data, labels, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_labels))

#%%
#%% Save and restore

#%% Weights only

# Save and load the weights of a model using tf.keras.Model.save_weights:

model = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(32,)),
            layers.Dense(10, activation='softmax')
            ])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Save weights to a TensorFlow Checkpoint file
model.save_weights('./weights/my_model')

# Restore the model's state,
# this requires a model with the same architecture.
model.load_weights('./weights/my_model')

# By default, this saves the model's weights in the TensorFlow `checkpoint` file format. 
# Weights can also be saved to the Keras HDF5 format 
# (the default for the multi-backend implementation of Keras):

# Save weights to a HDF5 file
model.save_weights('my_model.h5', save_format='h5')

# Restore the model's state
model.load_weights('my_model.h5')


#%%  Configuration only
"""
A model's configuration can be saved
— this serializes the model architecture without any weights. 
A saved configuration can recreate and initialize the same model, 
even without the code that defined the original model. 
Keras supports JSON and YAML serialization formats:
"""
# Serialize a model to JSON format
json_string = model.to_json()
json_string

import json
import pprint
pprint.pprint(json.loads(json_string))

#  Recreate the model (newly initialized) from the JSON:

fresh_model = tf.keras.models.model_from_json(json_string)

#%% Serializing a model to YAML format 
#   requires that you install pyyaml before you import TensorFlow:

yaml_string = model.to_yaml()
print(yaml_string)

# Recreate the model from the YAML:

fresh_model = tf.keras.models.model_from_yaml(yaml_string)
frash_model.summary()

"""
!!! Caution: Subclassed models are not serializable 
because their architecture is defined by the Python code in the body of the call method.
"""

#%%  Entire model
"""
The entire model can be saved to a file that contains the weight values, 
the model's configuration, and even the optimizer's configuration. 
This allows you to checkpoint a model and resume training later
— from the exact same state — without access to the original code.
"""

# Create a trivial model
model = tf.keras.Sequential([
  layers.Dense(10, activation='softmax', input_shape=(32,)),
  layers.Dense(10, activation='softmax')
])
    
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(data, labels, batch_size=32, epochs=5)

# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
model = tf.keras.models.load_model('my_model.h5')


#%% Eager execution
...

#%%
#%%  Distribution


#%%  Estimators
"""
The Estimators API is used for training models for distributed environments. 
This targets industry use cases such as distributed training on large datasets 
that can export a model for production.
"""

...

#%%  Multiple GPUs
