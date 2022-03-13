#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: Examining the TensorFlow Graph
part: 1 
subtitle:
version: 1.0
type: code chunks
keywords: [TensorBoard, Keras, MNIST]
description: Examples.
sources:
    - title: Examining the TensorFlow Graph
      link: https://www.tensorflow.org/tensorboard/graphs
      usage: copy
file:
    usage: 
        interactive: True
        terminal: False
    name: "04 - Model Graphs.py"
    path: "D:/ROBOCZY/Python/TensorBoard/"
    date: 2019-10-04
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

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."

"""
# Clear any logs from previous runs
!rm -rf ./logs/   (lin)

"""

#%% Define a Keras model
# In this example, the classifier is a simple four-layer Sequential model.

# Define the model.
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])

#%%
model.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 784)               0         28**2
_________________________________________________________________
dense (Dense)                (None, 32)                25120     (784 + 1)*32
_________________________________________________________________
dropout (Dropout)            (None, 32)                0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                330       (32 + 1)*10
=================================================================
Total params: 25,450
Trainable params: 25,450
Non-trainable params: 0
_________________________________________________________________
"""
#%% Download and prepare the training data.

(train_images, train_labels), (test_images, test_labels) = keras.datasets.fashion_mnist.load_data()
train_images = train_images / 255.0

#%% Train the model and log data
"""
Before training, define the Keras TensorBoard callback, specifying the log directory. 
By passing this callback to Model.fit(), you ensure that graph data is logged for visualization 
in TensorBoard.
"""

# Define the Keras TensorBoard callback.
logdir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

# Train the model.
model.fit(
    train_images,
    train_labels, 
    batch_size=64,
    epochs=5, 
    callbacks=[tensorboard_callback])

#%% Op-level graph
"""
Start TensorBoard and wait a few seconds for the UI to load. 
Select the Graphs dashboard by tapping “Graphs” at the top.

%tensorboard --logdir logs       

# doesn't work! (in Spyder at least...)
# go to anaconda prompt chdir to pwd
# run  tensorboard --logdir logs --host 127.0.0.1
# in browser go to 127.0.0.1:6006
"""
#%%
"""
By default, TensorBoard displays the op-level graph. 
(On the left, you can see the “Default” tag selected.) 
Note that the graph is inverted; data flows from bottom to top, 
so it’s upside down compared to the code. 
However, you can see that the graph closely matches the Keras model definition, 
with extra edges to other computation nodes.

Graphs are often very large, so you can manipulate the graph visualization:

    Scroll to zoom in and out
    Drag to pan
    Double clicking toggles node expansion (a node can be a container for other nodes)

You can also see metadata by clicking on a node. 
This allows you to see inputs, outputs, shapes and other details.
"""

#%%
#%% Graphs of tf.functions
"""
The examples so far have described graphs of Keras models, 
where the graphs have been created by defining Keras layers and calling Model.fit().

You may encounter a situation where you need to use the tf.function annotation to "autograph", 
i.e., transform, a Python computation function into a high-performance TensorFlow graph. 
For these situations, you use TensorFlow Summary Trace API to log autographed functions 
for visualization in TensorBoard.

To use the Sumary Trace API:

    Define and annotate a function with tf.function
    Use tf.summary.trace_on() immediately before your function call site.
    Add profile information (memory, CPU time) to graph by passing profiler=True
    With a Summary file writer, call tf.summary.trace_export() to save the log data

You can then use TensorBoard to see how your function behaves.
"""

#! it's better to restart the kernel !!!

# The function to be traced.
@tf.function
def my_func(x, y):
  # A simple hand-rolled layer.
  return tf.nn.relu(tf.matmul(x, y))

# Set up logging.
stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs\\func\\%s' % stamp
writer = tf.summary.create_file_writer(logdir)

# Sample data for your function.
x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

# Bracket the function call with  tf.summary.trace_on()  and  tf.summary.trace_export().
tf.summary.trace_on(graph=True, profiler=True)
# Call only one tf.function when tracing.
z = my_func(x, y)
with writer.as_default():
  tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=logdir)
  
#! anaconda prompt ... --logdir logs\func
  
#%%
""" You can now see the structure of your function as understood by TensorBoard. 
Click on the "Profile" radiobutton to see CPU and memory statistics.
"""