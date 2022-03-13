#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: Introduction to low level API
part: 1 
subtitle:
version: 1.0
type: code chunks
keywords: [TensorFlow, low level API]
description: examples 
sources:
    - title: Introduction to low level API
      link: https://www.tensorflow.org/guide/low_level_intro
      usage: copy
file:
    usage: 
        interactive: True
        terminal: False
    name: "01 - Introduction.py"
    path: "D:/ROBOCZY/Python/TensorFlow/Guide/Low Level API/"
    date: 2018-08-23
    authors:   
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: 
              - arkadiusz.kasprzyk@tieto.com
              - akasp666@google.com
              - arek@staart.pl      
"""  
#%%

cd "d:\ROBOCZY\Python\TensorFlow\Guide\Low Level API\"
pwd
ls

#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#%%
import numpy as np
import tensorflow as tf

#%%  Tensor Values
"""
The central unit of data in TensorFlow is the tensor. 
A tensor consists of a set of primitive values shaped into an array of any number of dimensions. 
A tensor's rank is its number of dimensions, 
while its shape is a tuple of integers specifying the array's length along each dimension. 
Here are some examples of tensor values:
"""
    
3.      # a rank 0 tensor; a scalar with shape [],

t1 = [1., 2., 3.] # a rank 1 tensor; a vector with shape [3]
t1
len(t1)
np.rank(t1)   # depricated! use:
np.ndim(t1)
np.shape(t1)

t2 = [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
len(t2)
np.ndim(t2)
np.shape(t2)

t3 = [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]
np.ndim(t3)
np.shape(t3)

"""
TensorFlow uses `numpy` arrays to represent tensor values.
"""

#%%  TensorFlow Core Walkthrough
"""
You might think of TensorFlow Core programs as consisting of two discrete sections:
- Building the computational graph (a `tf.Graph`).
- Running the computational graph (using a `tf.Session`).
"""

#%%  Graph
"""
A computational graph is a series of TensorFlow operations arranged into a graph. 
The graph is composed of two types of objects.
- `tf.Operation` (or "ops"): The nodes of the graph. 
  Operations describe calculations that consume and produce tensors.
- `tf.Tensor`: The edges in the graph. 
  These represent the values that will flow through the graph. 
  Most TensorFlow functions return `tf.Tensors`.

Important: `tf.Tensors` do not have values, 
   they are just __handles__ to elements in the computation graph.
"""

a = tf.constant(3., dtype=tf.float32)
b = tf.constant(4.,)
total = a + b
print(a)
print(b)
print(total)

"""
Each operation in a graph is given a unique name. 
This name is independent of the names the objects are assigned to in Python. 
Tensors are named after the operation that produces them followed by an output index, 
as in "add:0" above.
"""

#%%
# TensorBoard
"""
1. Open terminal (e.g. anaconda Prompt if you have Anaconda installed)
2. go to current directory, e.g. here: 
> cd "d:\ROBOCZY\Python\TensorFlow\Guide\Low Level API\"
3. delete first all the files  `events*`
> del events*
In these files old computational graphs are written.
You may leave them but they will interfere with your current graph, i.e.
you shoud have only one `event*` file in the directory -- the one with the current graph.
"""
# First you save the computation graph to a TensorBoard summary file
writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
#writer.flush()                                              #???

"""
This will produce an event file in the current directory with a name in the following format:
events.out.tfevents.{timestamp}.{hostname}

Now, in a new terminal, launch TensorBoard with the following shell command:
tensorboard --logdir ./ --host=127.0.0.1
(better use Anaconda Prompt)
in a browser: localhost:6006
"""

#%%

sess = tf.Session()
print(sess.run(total))

print(sess.run({'ab':(a, b), 'total':total}))

#%%
"""
During a call to tf.Session.run any tf.Tensor only has a single value. 
For example, the following code calls tf.random_uniform to produce a tf.Tensor 
that generates a random 3-element vector (with values in [0,1)):
"""
vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
"""
TB: if you refresh tensorboard view in your browser then you see that nothing changed,
no new elements were added to the graph view.
To see changes you must update the graph view with:
"""
writer.add_graph(tf.get_default_graph())

#%%
print(vec)
print(out1)
print(out2)
"""
The result shows a different random value on each call to run, 
but a consistent value during a single run (out1 and out2 receive the same random input):
"""
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

print(sess.run([out1, out2]))

"""
Some TensorFlow functions return `tf.Operations` instead of `tf.Tensors`. 
The result of calling run on an Operation is `None`. 

You run an `operation` to cause a side-effect, not to retrieve a value.   

Examples of this include the _initialization_, and _training_ ops demonstrated later.
"""

#%%  A placeholder is a promise to provide a value later, like a function argument.

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y

writer.add_graph(tf.get_default_graph())

print(sess.run(z, feed_dict={x: 3, y: 4.5}))
print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))

# The only difference between placeholders and other tf.Tensors is that
# placeholders throw an error if no value is fed to them

#%%  Datasets
"""
Placeholders work for simple experiments, 
but tf.data are the preferred method of _streaming_ data into a model.

To get a runnable `tf.Tensor` from a Dataset you must first convert it to a `tf.data.Iterator`, 
and then call the Iterator's `tf.data.Iterator.get_next()` method.

The simplest way to create an Iterator is with the 
`tf.data.Dataset.make_one_shot_iterator()` method. 
For example, in the following code the `next_item` tensor will return a row 
from the `my_data` array on each run call:
"""

my_data = [[0, 1], [2, 3], [4, 5], [6, 7]]
slices = tf.data.Dataset.from_tensor_slices(my_data)       #? .from_tensor_slices()         ???
next_item = slices.make_one_shot_iterator().get_next()     # depricated 
                                                           #! .make_one_shot_iterator()
                                                           #  .get_next() 
writer.add_graph(tf.get_default_graph())

#%%  Reaching the end of the data stream causes Dataset to throw an `OutOfRangeError`.

while True:
    try:
        print(sess.run(next_item))
    except tf.errors.OutOfRangeError:
        break

#%%
"""
If the Dataset depends on _stateful_ operations you may need
to _initialize_ the `iterator` before using it, as shown below:
"""

r = tf.random_normal([10, 3])                       #!! this is STATEFUL
dataset = tf.data.Dataset.from_tensor_slices(r)     #? .from_tensor_slices()         ???
iterator = dataset.make_initializable_iterator()    # depricated       
                                                    #! .make_initializable_iterator()
next_row = iterator.get_next()                      #  .get_next()
                                                            
writer.add_graph(tf.get_default_graph())

#%%
sess.run(iterator.initializer)                      #! iterator.initializer  
while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break


#%%  Layers

"""
A _trainable_ model must modify the values in the graph to get new outputs with the same input. 
`tf.layers` are the preferred way to add trainable parameters to a graph.
Layers package together both the `variables` and the `operations` that act on them.      ???

For example a _densely-connected_ layer performs a weighted sum across all inputs
for each output and applies an optional activation function.
The connection _weights_ and _biases_ are managed by the layer object.
"""

x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)
y = linear_model(x)                               # sth. depricated

writer.add_graph(tf.get_default_graph())
#%%n initializing layers

init = tf.global_variables_initializer()
sess.run(init)

'''
Calling `tf.global_variables_initializer()` only creates and returns
a _handle_ to a TensorFlow operation.
That _op_ will initialize all the global variables when we run it with
tf.Session.run.

Also note that this `global_variables_initializer()` only initializes variables
that existed in the graph when the initializer was created.
So the initializer should be one of the last things added during graph construction.
'''
# Executing layers
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}))

print(sess.run(y, {x: [1, 2, 3]}))      # ERROR
#%%  re-run

sess.run(init)
print(sess.run(y, {x: [[1, 2, 3]]}))
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}))

"""
Notice that only re-initialisation results in new weights within `layer` 
what gives new values to the output.
"""

#%% do not work!!! WHY???

sess.run(iterator.initializer)                      #! iterator.initializer  
while True:
    try:
        xx = next_row
        print(sess.run(y, {x: xx}))
    except tf.errors.OutOfRangeError:
        break


#%%  Layer Function Shortcuts
"""
For each layer class (like `tf.layers.Dense()`) 
TensorFlow also supplies a shortcut function (like `tf.layers.dense()`).
The shortcut function versions create and run the layer in a single call.
For example, the following code is equivalent to the earlier version:
"""

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)   # .dense(x, units)  instead of  .Dense(units)
                                  # depricated

writer.add_graph(tf.get_default_graph())

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))
print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]}))
'''
While convenient, this approach allows no access to the tf.layers.Layer object.
This makes introspection and debugging more difficult, and layer reuse impossible.
'''

#%%
# Feauture columns
"""
The easiest way to experiment with _feature columns_ is using the
tf.feature_column.input_layer() function.

This function only accepts _dense_ columns as inputs,                   # ?
so to view the result of a _categorical_ column you must wrap it in an
tf.feature_column.indicator_column().
For example:
"""

features = {
    'sales' : [[5], [10], [8], [9]],
    'department' : ['sports', 'sports', 'gardening', 'gardening']}

department_column = tf.feature_column.categorical_column_with_vocabulary_list(
    'department', ['sports', 'gardening'])

columns = [
    tf.feature_column.numeric_column('sales'),
    tf.feature_column.indicator_column(department_column)
    ]

inputs = tf.feature_column.input_layer(features, columns)            #!  .input_layer()
"""
IndicatorColumn._get_dense_tensor (from tensorflow.python.feature_column.feature_column_v2) 
is deprecated and will be removed in a future version.
Instructions for updating:
The old _FeatureColumn APIs are being deprecated. Please use the new FeatureColumn APIs instead.
"""

#%%
"""
Running the inputs tensor will parse the features into a batch of vectors

Feature columns can have _internal state_, like layers,
so they often need to be initialized.

Categorical columns use _lookup tables_ internally and these require
a separate initialization op,
`tf.tables_initializer()`.
"""

var_init   = tf.global_variables_initializer()
table_init = tf.tables_initializer()

sess = tf.Session()
sess.run((var_init, table_init))

"""
Once the _internal state_ has been initialized you can run inputs like any other tf.Tensor:
"""
print(sess.run(inputs))

writer.add_graph(tf.get_default_graph())

#%%
# Training

#%%  define the data

x      = tf.constant([[1],  [2],  [3],  [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

#%%  define the model

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)

#%%
# you can evaluate the prediction as follows

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(y_pred))

#%%
# Loss
"""
To optimize a model, you first need to define the loss.
We'll use the mean square error, a standard loss for regression problems.

While you could do this manually with lower level math operations,
the tf.losses module provides a set of common loss functions.
You can use it to calculate the mean square error as follows:
"""
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

print(sess.run(loss))

#%%  Training
"""
TensorFlow provides _optimizers_ implementing standard optimization algorithms.
These are implemented as sub-classes of `tf.train.Optimizer`. 
They incrementally change each variable in order to minimize the loss. 
The simplest optimization algorithm is gradient descent, 
implemented by `tf.train.GradientDescentOptimizer()`. 
It modifies each variable according to the magnitude of the derivative of loss 
with respect to that variable. For example:
"""

optimizer = tf.train.GradientDescentOptimizer(.01)
train = optimizer.minimize(loss)

"""
Since train is an op, not a tensor, it doesn't return a value when run.
To see the progression of the loss during training, we run the loss tensor
at the same time, producing output like the following:
"""

for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)

writer.add_graph(tf.get_default_graph())

#%%
# complete program

%reset

#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#%%

x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())        #! do it frequently and see on the tensorboard what's going on

linear_model = tf.layers.Dense(units=1)

y_pred = linear_model(x)
"""
calling VarianceScaling.__init__ (from tensorflow.python.ops.init_ops) 
with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Call initializer instance with the dtype argument instead of passing it to the constructor
"""
loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
optimizer = tf.train.GradientDescentOptimizer(.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for i in range(100):
    _, loss_value = sess.run((train, loss))
    print(loss_value)

print(sess.run(y_pred))
print(sess.run(y_true))


#%%


