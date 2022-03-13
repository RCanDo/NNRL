#! python3
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Quickstart For Experts
part: 1
subtitle:
version: 1.0
type: code chunks
keywords: [TensorFlow, quickstart, experts, advanced]
description: Examples.
sources:
    - title: TensorFlow 2 quickstart for experts
      link: https://www.tensorflow.org/tutorials/quickstart/advanced/
      usage: copy
file:
    usage:
        interactive: True
        terminal: False
    name: "02 - Experts.py"
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

# from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

#%% Load and prepare the MNIST dataset.

mnist = tf.keras.datasets.mnist
type(mnist)   # module
dir(mnist)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
type(x_train)  # numpy.ndarray
x_train.shape  # (60000, 28, 28)
x_train[1, :, :]

# normalize
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train[1, :, :]

#%%
import matplotlib.pyplot as plt
plt.imshow(x_train[2], cmap='Greys', interpolation='nearest')
plt.show()

#%% Add a channels dimension
help(tf.newaxis)
x_train = x_train[..., tf.newaxis]                                         #!!! the same as arr[..., np.newaxis]
x_train.shape   # (60000, 28, 28, 1)

x_test = x_test[..., tf.newaxis]

#%% Use tf.data to batch and shuffle the dataset:

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
type(train_ds)   # tensorflow.python.data.ops.dataset_ops.TensorSliceDataset

train_ds = train_ds.shuffle(10000)
type(train_ds)   # tensorflow.python.data.ops.dataset_ops.ShuffleDataset

train_ds = train_ds.batch(32)
type(train_ds)   # tensorflow.python.data.ops.dataset_ops.BatchDataset

#%%
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)
type(test_ds)    # tensorflow.python.data.ops.dataset_ops.BatchDataset

#%% Build the tf.keras model using the Keras model subclassing API:

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(filters=32, kernel_size=3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10, activation='softmax')

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

#%% Create an instance of the model

model = MyModel()
dir(model)
model.summary()
"""ValueError: This model has not yet been built.
Build the model first by calling `build()` or calling `fit()` with some data,
or specify an `input_shape` argument in the first layer(s) for automatic build.
"""

#%% Choose an _optimizer_ and _loss function_ for training:

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
type(loss_object)  # tensorflow.python.keras.losses.SparseCategoricalCrossentropy
dir(loss_object)
loss_object.name   # 'sparse_categorical_crossentropy'

optimizer = tf.keras.optimizers.Adam()
type(optimizer)    # tensorflow.python.keras.optimizer_v2.adam.Adam
dir(optimizer)
optimizer.weights    # []
optimizer.variables  # <bound method OptimizerV2.variables of <tensorflow.python.keras.optimizer_v2.adam.Adam
optimizer.variables()  # []

#%% Select _metrics_ to measure the _loss_ and the _accuracy_ of the model.
# These metrics accumulate the values over epochs and then print the overall result.

train_loss     = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss      = tf.keras.metrics.Mean(name='test_loss')
test_accuracy  = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

#%% Use tf.GradientTape to train the model:

@tf.function
def train_step(images, labels):

    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

#%% alternatively (check it!)
"""
optimizer.minimize(loss, vars_list) ==    tf.GradientTape()
                                       -> tape.gradient(loss, vars_list)
                                       -> optimizer.apply_gradents(zip(gradients, vars_list))
"""
@tf.function
def train_step_v2(images, labels):
  predictions = model(images)
  loss = loss_object(labels, predictions)
  optimizer.minimize(loss, model.trainable_variables)

  train_loss(loss)
  train_accuracy(labels, predictions)

#%% Test the model:

@tf.function
def test_step(images, labels):
    # Assumes that model is already trained
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

#%%

EPOCHS = 5

for epoch in range(EPOCHS):
  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

# The image classifier is now trained to ~98% accuracy on this dataset.

#%%

model.summary()
"""
Model: "my_model_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d_2 (Conv2D)            multiple                  320       = (3*3*1 + 1)*32   1 bias, 1 channel
_________________________________________________________________
flatten_1 (Flatten)          multiple                  0         -> 32* 26**2 = 21632  (smaller "pics" as 0 padding above)
_________________________________________________________________
dense_6 (Dense)              multiple                  2769024   = (21632 + 1)*128
_________________________________________________________________
dense_7 (Dense)              multiple                  1290      = (128 + 1)*10
=================================================================
Total params: 2,770,634
Trainable params: 2,770,634
Non-trainable params: 0
_________________________________________________________________
"""