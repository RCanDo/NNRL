#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: TensorBoard Scalars
part: 1 
subtitle:
version: 1.0
type: code chunks
keywords: [TensorBoard, Keras, callbacks, learning rate scheduler, scalars]
description: Examples.
sources:
    - title: TensorBoard Scalars: Logging training metrics in Keras
      link: https://www.tensorflow.org/tensorboard/scalars_and_keras
      usage: copy
file:
    usage: 
        interactive: True
        terminal: False
    name: "02 - Scalars And Metrics.py"
    path: "D:/ROBOCZY/Python/TensorBoard/"
    date: 2019-10-10
    authors:   
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: 
              - arkadiusz.kasprzyk@tieto.com
              - akasp666@google.com
              - arek@staart.pl      
"""  
#%%
cd "D:/ROBOCZY/Python/TensorBoard/"
pwd

#%%
"""
Overview

Machine learning invariably involves understanding key metrics such as loss 
and how they change as training progresses. 
These metrics can help you understand if you're overfitting, 
for example, or if you're unnecessarily training for too long. 
You may want to compare these metrics across different training runs 
to help debug and improve your model.

TensorBoard's Scalars Dashboard allows you to visualize these metrics 
using a simple API with very little effort. 
This tutorial presents very basic examples to help you learn how to use these APIs 
with TensorBoard when developing your Keras model. 
You will learn how to use the Keras TensorBoard callback 
and TensorFlow Summary APIs to visualize default and custom scalars.
"""

#%% Setup
"""
# Ensure TensorFlow 2.0 is installed.
!pip install -q tf-nightly-2.0-preview
# Load the TensorBoard notebook extension.
%load_ext tensorboard
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

import numpy as np

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

# TensorFlow version:  2.0.0-dev20190226

#%% Set up data for a simple regression
"""
You're now going to use Keras to calculate a regression, i.e., 
find the best line of fit for a paired data set. 
(While using neural networks and gradient descent is overkill for this kind of problem, 
it does make for a very easy to understand example.)

You're going to use TensorBoard to observe how training and test loss change across epochs. 
Hopefully, you'll see training and test loss decrease over time and then remain steady.

First, generate 1000 data points roughly along the line y = 0.5x + 2. 
Split these data points into training and test sets. 
Your hope is that the neural net learns this relationship.
"""
data_size = 1000
# 80% of the data is for training.
train_pct = 0.8

train_size = int(data_size * train_pct)

# Create some input data between -1 and 1 and randomize it.
x = np.linspace(-1, 1, data_size)
np.random.shuffle(x)

# Generate the output data.
# y = 0.5x + 2 + noise
y = 0.5 * x + 2 + np.random.normal(0, 0.05, (data_size, ))

# Split into test and train pairs.
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

#%% Training the model and logging loss
"""
You're now ready to define, train and evaluate your model.

To log the loss scalar as you train, you'll do the following:

    Create the Keras TensorBoard callback
    Specify a log directory
    Pass the TensorBoard callback to Keras' Model.fit().

TensorBoard reads log data from the log directory hierarchy. 
In this notebook, the root log directory is logs/scalars, 
suffixed by a timestamped subdirectory. 
The timestamped subdirectory enables you to easily identify and select training runs 
as you use TensorBoard and iterate on your model.
"""
#!
logdir = "logs\\scalars\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
#!

model = keras.models.Sequential([
    keras.layers.Dense(16, input_dim=1),
    keras.layers.Dense(1),
])

model.compile(
    loss='mse',   # keras.losses.mean_squared_error
    optimizer=keras.optimizers.SGD(lr=0.2),
)

print("Training ... With default parameters, this takes less than 10 seconds.")
training_history = model.fit(
    x_train, # input
    y_train, # output
    batch_size=train_size,
    verbose=0, # Suppress chatty output; use Tensorboard instead
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback],     #!
)

print("Average test loss: ", np.average(training_history.history['loss']))

# Training ... With default parameters, this takes less than 10 seconds.
# Average test loss:  0.05271831926424056

#%% Examining loss using TensorBoard
"""
Now, start TensorBoard, specifying the root log directory you used above.
Wait a few seconds for TensorBoard's UI to spin up.
%tensorboard --logdir logs\\scalars --host 127.0.0.1
(it's better to use terminal like 'anaconda prompt')

You may see TensorBoard display the message "No dashboards are active for the current data set". 
That's because initial logging data hasn't been saved yet. 
As training progresses, the Keras model will start logging data. 
TensorBoard will periodically refresh and show you your scalar metrics. 
If you're impatient, you can tap the Refresh arrow at the top right.

As you watch the training progress, note how both training and validation loss rapidly decrease, 
and then remain stable. 
In fact, you could have stopped training after 25 epochs, 
because the training didn't improve much after that point.

Hover over the graph to see specific data points. 
You can also try zooming in with your mouse, or selecting part of them to view more detail.

Notice the "Runs" selector on the left. 
A "run" represents a set of logs from a round of training, 
in this case the result of Model.fit(). 
Developers typically have many, many runs, as they experiment and develop their model over time.

Use the Runs selector to choose specific runs, or choose from only training or validation. 
Comparing runs will help you evaluate which version of your code is solving your problem better.

Ok, TensorBoard's loss graph demonstrates that the loss consistently decreased 
for both training and validation and then stabilized. 
That means that the model's metrics are likely very good! 
Now see how the model actually behaves in real life.

Given the input data (60, 25, 2), 
the line y = 0.5x + 2 should yield (32, 14.5, 3). 
Does the model agree?
"""
print(model.predict([60, 25, 2]))
# True values to compare predictions against: 
# [[32.0]
#  [14.5]
#  [ 3.0]]
"""
[[31.893326]
 [14.458023]
 [ 3.000537]]
...
Not bad!
"""
#%%Add some figure!!!
from matplotlib import pyplot as plt

def my_plot(model):
    plt.figure()
    plt.scatter(x_train, y_train, marker='.')
    plt.plot(x_test, model.predict(x_test), c='red')
    plt.scatter(x_test, y_test, marker="+")

my_plot(model)

model.trainable_variables     # no possible interpretation...

#%% extremaly primitive model

# let's do sth very primitive
logdir = "logs\\scalars\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

model02 = keras.models.Sequential([
    keras.layers.Dense(1, input_dim=1),
])

model02.compile(
    loss='mse',   # keras.losses.mean_squared_error
    optimizer=keras.optimizers.SGD(lr=0.2),
)

history02 = model02.fit(x_train, y_train, 
                      batch_size=1, verbose=0, epochs=100,
                      validation_data=(x_test, y_test),
                      callbacks=[tensorboard_callback])

my_plot(model02)

"""
see now TB; there is huge variability in such a primitive model
maybe because of lr=.2 ???
"""


#%% Logging custom scalars
"""
What if you want to log custom values, such as a dynamic learning rate? 
To do that, you need to use the TensorFlow Summary API.

Retrain the regression model and log a custom learning rate. Here's how:

1. Create a file writer, using tf.summary.create_file_writer().
2. Define a custom learning rate function. 
   This will be passed to the Keras LearningRateScheduler callback.
3. Inside the learning rate function, use tf.summary.scalar() to log the custom learning rate.
    Pass the LearningRateScheduler callback to Model.fit().

In general, to log a custom scalar, you need to use tf.summary.scalar() with a file writer. 
The file writer is responsible for writing data for this run to the specified directory and is implicitly used when you use the tf.summary.scalar().
"""
#%%
logdir = "logs\\scalars\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "\\metrics")

file_writer.set_as_default()    #!!!

def lr_schedule(epoch):
  """
  Returns a custom learning rate that decreases as epochs progress.
  """
  learning_rate = 0.2
  if epoch > 10:
    learning_rate = 0.02
  if epoch > 20:
    learning_rate = 0.01
  if epoch > 50:
    learning_rate = 0.005

  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)   #!!!
  return learning_rate

#%%
lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)    #!
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

#%%
model10 = keras.models.Sequential([
    keras.layers.Dense(16, input_dim=1),
    keras.layers.Dense(1),
])

model10.compile(
    loss='mse', # keras.losses.mean_squared_error
    optimizer=keras.optimizers.SGD(), # now `lr` is not setup as it will be providede by lr_schedule()
)

history10 = model10.fit(
    x_train, # input
    y_train, # output
    batch_size=train_size,
    verbose=0, # Suppress chatty output; use Tensorboard instead
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[tensorboard_callback, lr_callback],
)

#%% Let's look at TensorBoard again.
"""
%tensorboard --logdir logs/scalars

Using the "Runs" selector on the left, notice that you have a <timestamp>/metrics run. 
Selecting this run displays a "learning rate" graph that allows you to verify 
the progression of the learning rate during this run.

You can also compare this run's training and validation loss curves against your earlier runs.

How does this model do?
"""
print(model10.predict([60, 25, 2]))
# True values to compare predictions against: 
# [[32.0]
#  [14.5]
#  [ 3.0]]
"""
[[31.89224  ]
 [14.457571 ]
 [ 3.0005028]]
"""
my_plot(model10)

#%% 