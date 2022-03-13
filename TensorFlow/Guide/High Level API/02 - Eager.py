# -*- coding: utf-8 -*-
"""
file: 01-keras.py 
title: 
chapter: 
book: 
link: https://www.tensorflow.org/guide/keras
date: 2019-02-25 Mon 10:22:35
author: kasprark
email: arkadiusz.kasprzyk@tieto.com
"""

"""
...

Eager execution supports most TensorFlow operations and GPU acceleration. 
For a collection of examples running in eager execution, see: 
    tensorflow/contrib/eager/python/examples.

...
"""

#%%

# !!! RESTART KERNEL FIRST !!!

cd "D:\ROBOCZY\Python\TensorFlow\Guide\HighLevelAPI"
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

tf.enable_eager_execution()   # !!! RESTART KERNEL FIRST !!!


#%% Now you can run TensorFlow operations and the results will return immediately: 

tf.executing_eagerly() 

#%%

x = [[2.]]
m = tf.matmul(x, x)
print("hello, {}".format(m))

#%%

a = tf.constant([[1, 2],
                 [3, 4]])
print(a)
a.eval()        # NotImplementedError: eval is not supported when eager execution is enabled, 
                # is .numpy() what you're looking for?
a.numpy()       # OK!!!

#%%

# Broadcasting support
b = tf.add(a, 1)
print(b)

# Operator overloading is supported
c = a * b
c

#%%
# Use NumPy values
import numpy as np

c = np.multiply(a, b)
c

# Obtain numpy value from a tensor:
a.numpy()

"""
The tf.contrib.eager module contains symbols available to both eager and graph execution 
environments and is useful for writing code to work with graphs:
"""

tfe = tf.contrib.eager
tfe

#%%
#%%  Dynamic control flow
"""
A major benefit of eager execution is that all the functionality of the host language
is available while your model is executing.
So, for example, it is easy to write fizzbuzz:
"""

def fizzbuzz(max_num):
  counter = tf.constant(0)
  max_num = tf.convert_to_tensor(max_num)
  for num in range(1, max_num.numpy()+1):
    num = tf.constant(num)
    if int(num % 3) == 0 and int(num % 5) == 0:
      print('FizzBuzz')
    elif int(num % 3) == 0:
      print('Fizz')
    elif int(num % 5) == 0:
      print('Buzz')
    else:
      print(num.numpy())
    counter += 1
    
fizzbuzz(15)


#%%
#%%  Build a model
"""
Many machine learning models are represented by composing layers. 
When using TensorFlow with eager execution you can either write your own layers 
or use a layer provided in the tf.keras.layers package.

While you can use any Python object to represent a layer, 
TensorFlow has tf.keras.layers.Layer as a convenient base class. 
Inherit from it to implement your own layer:
"""

class MySimpleLayer(tf.keras.layers.Layer):
    
  def __init__(self, output_units):
    super(MySimpleLayer, self).__init__()
    self.output_units = output_units

  def build(self, input_shape):
    # The build method gets called the first time your layer is used.
    # Creating variables on build() allows you to make their shape depend
    # on the input shape and hence removes the need for the user to specify
    # full shapes. It is possible to create variables during __init__() if
    # you already know their full shapes.
    self.kernel = self.add_variable(
      "kernel", [input_shape[-1], self.output_units])

  def call(self, input):
    # Override call() instead of __call__ so we can perform some bookkeeping.
    return tf.matmul(input, self.kernel)


my_layer = MySimpleLayer(5)
my_layer


data = tf.random_uniform([100,10])
data
my_layer(data)   # ! TypeError: unsupported operand type(s) for /: 'Dimension' and 'float'


#%%
#%% !!!

model = tf.keras.Sequential([
          tf.keras.layers.Dense(10, input_shape=(10,)),  # must declare input shape
          tf.keras.layers.Dense(10)
        ])

model.summary()
model.weights

data = tf.random_uniform((20,10))
data

data = np.random.random((20,10))
data

res = model(data)
res
res.numpy() 

res = model(data[:1])
res
res.numpy() 

pred = model.predict(data)
pred

pred = model.predict(data[0])

   
#%%    
#%%

class MNISTModel(tf.keras.Model):
  def __init__(self):
    super(MNISTModel, self).__init__()
    self.dense1 = tf.keras.layers.Dense(units=10)
    self.dense2 = tf.keras.layers.Dense(units=10)

  def call(self, input):
    """Run the model."""
    result = self.dense1(input)
    result = self.dense2(result)
    result = self.dense2(result)  # reuse variables from dense2 layer
    return result

model = MNISTModel()

"""
It's not required to set an input shape for the tf.keras.Model class 
since the parameters are set the first time input is passed to the layer.
"""

#%%
#%%  Eager training

#%%  Computing gradients

w = tf.Variable([[1.0]])
with tf.GradientTape() as tape:
  loss = w * w

grad = tape.gradient(loss, w)
print(grad)  # => tf.Tensor([[ 2.]], shape=(1, 1), dtype=float32)





