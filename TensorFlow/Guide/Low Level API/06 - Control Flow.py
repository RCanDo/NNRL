# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 12:51:36 2019

@author: kasprark
"""

# -*- coding: utf-8 -*-
"""
file: 06-control_flow.py
title: 
chapter: AutoGraph: Easy control flow for graphs
book: 
link: https://www.tensorflow.org/guide/autograph
date: 2019-02-25 Mon 10:22:35
author: kasprark
email: arkadiusz.kasprzyk@tieto.com
"""

%reset

#%%

cd "D:\ROBOCZY\Python\TensorFlow\Guide\HighLevelAPI"
pwd
ls

#%%
#%%  Setup

from __future__ import division, print_function, absolute_import

import tensorflow as tf
layers = tf.keras.layers
from tensorflow import contrib
autograph = contrib.autograph

import numpy as np
import matplotlib.pyplot as plt


#%%  We'll enable eager execution for demonstration purposes,
#    but AutoGraph works in both eager and graph execution environments:

tf.enable_eager_execution()

"""
Note: AutoGraph converted code is designed to run during graph execution. 
When eager exectuon is enabled, use explicit graphs (as this example shows) or 
tf.contrib.eager.defun.
"""

#%%
#%%  Automatically convert Python control flow
"""
AutoGraph will convert much of the Python language 
into the equivalent TensorFlow graph building code.

Note: In real applications batching is essential for performance. 
The best code to convert to AutoGraph is code where the control flow is decided at the batch level.
If making decisions at the individual example level, 
you must index and batch the examples to maintain performance while applying the control flow logic. 
"""


# AutoGraph converts a function like:

def square_if_positive(x):
  if x > 0:
    x = x * x
  else:
    x = 0.0
  return x

# to a function that uses graph building:

print(autograph.to_code(square_if_positive))

print('Eager results: %2.2f, %2.2f' % (square_if_positive(tf.constant(9.0)), 
                                       square_if_positive(tf.constant(-9.0))))

# Generate a graph-version and call it:

tf_square_if_positive = autograph.to_graph(square_if_positive)

with tf.Graph().as_default():  
  # The result works like a regular op: takes tensors in, returns tensors.
  # You can inspect the graph using tf.get_default_graph().as_graph_def()
  g_out1 = tf_square_if_positive(tf.constant( 9.0))
  g_out2 = tf_square_if_positive(tf.constant(-9.0))
  with tf.Session() as sess:
    print('Graph results: %2.2f, %2.2f\n' % (sess.run(g_out1), sess.run(g_out2)))


#%%
"""
AutoGraph supports common Python statements like while, for, if, break, and return, 
with support for nesting. 

Compare this function with the complicated graph verson displayed in the following code blocks:
"""

# Continue in a loop
def sum_even(items):
  s = 0
  for c in items:
    if c % 2 > 0:
      continue
    s += c
  return s

print('Eager result: %d' % sum_even(tf.constant([10,12,15,20])))

tf_sum_even = autograph.to_graph(sum_even)

with tf.Graph().as_default(), tf.Session() as sess:
    print('Graph result: %d\n\n' % sess.run(tf_sum_even(tf.constant([10,12,15,20]))))

print(autograph.to_code(sum_even))


#%%
#%%  Decorator

@autograph.convert()
def fizzbuzz(i, n):
  while i < n:
    msg = ''
    if i % 3 == 0:
      msg += 'Fizz'
    if i % 5 == 0:
      msg += 'Buzz'
    #if msg == '':
    #  msg = tf.as_string(i)
    print(msg)
    i += 1
  return i

with tf.Graph().as_default():
  final_i = fizzbuzz(tf.constant(10), tf.constant(16))
  # The result works like a regular op: takes tensors in, returns tensors.
  # You can inspect the graph using tf.get_default_graph().as_graph_def()
  with tf.Session() as sess:
    sess.run(final_i)

# !!! ERROR!!!!
    
#fizzbuzz(10, 16)

#%%

def fizzbuzz(i, n):
  while i < n:
    msg = ''
    if i % 3 == 0:
      msg += 'Fizz'
    if i % 5 == 0:
      msg += 'Buzz'
    if msg == '':
      msg = tf.as_string(i.dtype("int32"))
    print(msg)
    i += 1
  return i

tf_fizzbuzz = autograph.to_graph(fizzbuzz)    

with tf.Graph().as_default(), tf.Session() as sess:
    sess.run(tf_fizzbuzz(tf.constant(10), tf.constant(16)))
    
# !!! ERROR !!!!
    


#%%
#%%  Examples

#%%  Assert

@autograph.convert()
def inverse(x):
  assert x != 0.0, 'Do not pass zero!'
  return 1.0 / x

with tf.Graph().as_default(), tf.Session() as sess:
  try:
    print(sess.run(inverse(tf.constant(0.0))))
  except tf.errors.InvalidArgumentError as e:
    print('Got error message:\n    %s' % e.message)


#%%  Print
    
@autograph.convert()
def count(n):
  i=0
  while i < n:
    print(i)
    i += 1
  return n
    
with tf.Graph().as_default(), tf.Session() as sess:
    print(sess.run(count(tf.constant(5))))
    

#%%  Lists

@autograph.convert()
def arange(n):
  z = []
  # We ask you to tell us the element dtype of the list
  autograph.set_element_type(z, tf.int32)
  
  for i in tf.range(n):
    z.append(i)
  # when you're done with the list, stack it
  # (this is just like np.stack)
  return autograph.stack(z) 


with tf.Graph().as_default(), tf.Session() as sess:
    print(sess.run(arange(tf.constant(10))))


#%%  Nested control flow
    
@autograph.convert()
def nearest_odd_square(x):
  if x > 0:
    x = x * x
    if x % 2 == 0:
      x = x + 1
  return x

with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(nearest_odd_square(tf.constant(4))))
    print(sess.run(nearest_odd_square(tf.constant(5))))
    print(sess.run(nearest_odd_square(tf.constant(6))))


#%%  While loop

@autograph.convert()
def square_until_stop(x, y):
  while x < y:
    x = x * x
  return x
    
with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(square_until_stop(tf.constant(4), tf.constant(100))))


#%%  For loop
    
@autograph.convert()
def squares(nums):

  result = []
  autograph.set_element_type(result, tf.int32)   ## ERROR for tf.int64

  for num in nums: 
    result.append(num * num)
    
  return autograph.stack(result)

    
with tf.Graph().as_default():  
  with tf.Session() as sess:
    print(sess.run(squares(tf.constant(np.arange(10)))))


#%%  Break

@autograph.convert()
def argwhere_cumsum(x, threshold):
  current_sum = 0.0
  idx = 0
  for i in tf.range(len(x)):
    idx = i
    if current_sum >= threshold:
      break
    current_sum += x[i]
  return idx

N = 10
with tf.Graph().as_default():  
  with tf.Session() as sess:
    idx = argwhere_cumsum(tf.ones(N), tf.constant(float(N/2)))
    print(sess.run(idx))


#%%
#%%  Interoperation with tf.Keras

"""
For stateless functions, like collatz shown below, the easiest way to include them in a keras model 
is to wrap them up as a layer using tf.keras.layers.Lambda.
"""

import numpy as np


#%%  Stateless functions

@autograph.convert()
def collatz(x):
  x = tf.reshape(x,())
  assert x > 0
  n = tf.convert_to_tensor((0,)) 
  while x!=1:
    n += 1
    if x%2 == 0:
      x = x // 2
    else:
      x = 3 * x + 1
      
  return n


with tf.Graph().as_default():
  model = tf.keras.Sequential([
    tf.keras.layers.Lambda(collatz, input_shape=(1,), output_shape=())
  ])
  
  result = model.predict(np.array([6171]))
  print(result)


#%%  Custom Layers and Models
  


