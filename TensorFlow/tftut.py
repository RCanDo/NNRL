# -*- coding: utf-8 -*-
"""
title: TensorFlow Tutorial
    book: Bharath Ramsundar - TensorFlow Tutorial (44sl)
        file: "Ramsundar - TensorFlow Tutorial (44sl).pdf"
    chapter:
author: Arkadiusz Kasprzyk
email: akasp@interia.pl; arkadiusz.kasprzyk@tieto.com
date: Sat Jun  9 10:47:50 2018
"""

#%% numpy

import numpy as np

a = np.zeros((2, 2))
a
b = np.zeros((2, 2))
b

np.sum(b, axis=1)   #array([2., 2.])
a.shape
np.reshape(a, (1, 4))
a

#%%

import tensorflow as tf

tf.InteractiveSession()

a = tf.zeros((2, 2))
a
a.eval()

b = tf.ones((2, 2))
b
b.eval()

tf.reduce_sum(b, reduction_indices=1).eval()

a.get_shape()

tf.reshape(a, (1, 4)).eval()

a[0, 0]
a[0, 0].eval()
a[1,:].eval()

#%%
#  TensorFlow requires explicit evaluation!
#%%

print(a)
print(a.eval())

#%%
"""
TensorFlow Session Object (1)
“A Session object encapsulates the environment in which Tensor objects are evaluated”
"""

a = tf.constant(5.)
a
a.eval()

b = tf.constant(6.)
c = a * b

c
c.eval()

with tf.Session() as ss:
    print(ss.run(c))
    print(c.eval())             # c.eval() is syntactic sugar for ss.run(c) in the currently active session
    
#%%
"""
TensorFlow Session Object (2)
● tf.InteractiveSession() is just convenient syntactic sugar for keeping a default session open in ipython.

● sess.run(c) is an example of a TensorFlow Fetch.

Tensorflow Computation Graph
● “TensorFlow programs are usually structured into a construction phase, that assembles a graph, 
and an execution phase that uses a session to execute ops in the graph.”

● All computations add nodes to global default graph

● “TensorFlow programs are usually structured into a construction phase, that assembles a graph, 
and an execution phase that uses a session to execute ops in the graph.”

● All computations add nodes to global default graph
"""

#%%
"""
TensorFlow Variables

● “When you train a model you use variables to hold and update parameters. 
Variables are in-memory buffers containing tensors”

● All tensors we’ve used previously have been constant tensors, not variables
"""

w1 = tf.ones((2, 2))
w1
w1.eval()

with tf.Session() as ss:
    w1run = ss.run(w1)
    print(w1run)
    
w1run
print(w1run)

w1.eval()
print(w1.eval())

#%% 
"""
TensorFlow variables must be initialized before they have values! 
Contrast with constant tensors.
"""

w2 = tf.Variable(w1, name="wights")

with tf.Session() as ss:
    ss.run(tf.initialize_all_variables())   # !!!
    print(ss.run(w2))

#%%
"""
Variable objects can be initialized from constants or random values
"""

ww = tf.Variable(tf.zeros((2, 2)), name="weights")

rr = tf.Variable(tf.random_normal((2, 2)), name="random_wieghts")

with tf.Session() as ss:
    ss.run(tf.initialize_all_variables())
    print(ss.run(ww))
    print(ss.run(rr))
    
#%%
"""Updating variable state
"""

state = tf.Variable(0, name="counter")
new_value = tf.add(state, tf.constant(1))
update = tf.assign(state, new_value)

with tf.Session() as ss:
    ss.run(tf.initialize_all_variables())
    print(ss.run(state))        # 0
    print(ss.run(update))       # 1
    for _ in range(3):
        print(ss.run(state))    # 1, ..., 3
        print(ss.run(update))   # 2, ..., 4   i.e. run(updatge) returns the state after update 

# run it twice ! to see that sessions are separated
    
#%%
"""
Fetching Variable State
"""

input1 = tf.constant(3.)
input2 = tf.constant(2.)
input3 = tf.constant(5.)

intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)

with tf.Session() as ss:
    result = ss.run([mul, intermed])
    print(result)
    
#%%
"""
Inputting Data
● All previous examples have manually defined tensors.
How can we input external data into TensorFlow?
● Simple solution: Import from Numpy
"""

a = np.zeros((3, 3))
a
ta = tf.convert_to_tensor(a)
ta
ta.eval()

"""
Placeholders and Feed Dictionaries (1)
● Inputting data with tf.convert_to_tensor() is convenient, but doesn’t scale.
● Use tf.placeholder variables (dummy nodes that provide entry points for data to computational graph).
● A feed_dict is a python dictionary mapping from tf.placeholder vars 
(or their names) to data (numpy arrays, lists, etc.).
"""

input1 = tf.placeholder(tf.float32)
input1
type(input1)    # tensorflow.python.framework.ops.Tensor
input1.eval()   # NO !!! no .eval() method for  ..

input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)
output
type(output)    # tensorflow.python.framework.ops.Tensor
output.eval()   # NO !!! no .eval() method for  ..
        
with tf.Session() as ss:
    out = ss.run([output], feed_dict={input1:[7.], input2:[2.]})
    print(out)
    print(type(out))    # list !
    
#%%
"""    
Variable Scope
● Complicated TensorFlow models can have hundreds of variables.
    ○ tf.variable_scope() provides simple name-spacing to avoid clashes.
    ○ tf.get_variable() creates/accesses variables from within a variable scope.
● Variable scope is a simple type of namespacing that adds prefixes to variable names within scope
"""

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v = tf.get_variable("v", [1])

v
v.name
assert v.name == "foo/bar/v:0"

"""
● Variable scopes control variable (re)use
"""

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])
    tf.get_variable_scope().reuse_variables()
    v1 = tf.get_variable("v", [1])

v
v.eval()    # ERROR: Attempting to use uninitialized value foo/v
v1
assert v1 == v

"""
● You’ll need to use reuse_variables() to implement RNNs in homework
"""

#%%
"""
Understanding get_variable
● Behavior depends on whether variable reuse enabled
● Case 1: reuse set to false
    ○ Create and return new variable
"""

with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])    # ValueError: Variable foo/v already exists !!!
    


with tf.variable_scope("foo"):
    w = tf.get_variable("w", [1])
    
assert w.name == "foo/w:0"

"""
● Case 2: Variable reuse set to true
    ○ Search for existing variable with given name. Raise ValueError if none found.
"""

with tf.variable_scope("foo"):
    z = tf.get_variable("z", [1])

with tf.variable_scope("foo", reuse=True):
    z1 = tf.get_variable("z", [1])

assert z == z1

#%%
#%%
"""
Linear Regression in TensorFlow
"""

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb


# Define input data
xx = np.arange(100, step=.1)
yy = xx + 20 * np.sin(xx/10)

# Plot input data
plt.scatter(xx, yy)

#%%

# Define data size and batch size
n = len(xx)
batch_size = 100

# TensorFlow is finickly about shapes, so resize
xx = np.reshape(xx, (n, 1))
xx
yy = np.reshape(yy, (n, 1))
yy

# Define placeholders for input

x = tf.placeholder(tf.float32, shape=(batch_size, 1))
y = tf.placeholder(tf.float32, shape=(batch_size, 1))

# define variables to be learned
with tf.variable_scope("linear_regression"):
    w = tf.get_variable("weights", (1, 1), initializer=tf.random_normal_initializer())
        # note  reuse=False  so there tensors are created anew 
    b = tf.get_variable("bias", (1,), initializer=tf.constant_initializer(0.))
    yhat = tf.matmul(x, w) + b
    loss = tf.reduce_sum((y - yhat)**2/n)

yhat
loss

#%%
"""sample code to run one-step of gradient descent
"""
opt = tf.train.AdamOptimizer()
opt_operation = opt.minimize(loss) # TensorFlow scope is not Python scope!
                                    # Python variable loss is still visible

with tf.Session() as ss:
    ss.run(tf.global_variables_initializer())    # old:  tf.initialize_all_variables() -- depricated
    indices = np.random.choice(n, batch_size)
    x_batch, y_batch = xx[indices], yy[indices]
    ss.run([opt_operation], feed_dict={x: x_batch, y: y_batch})
        # How does this actually work under the hood?
        # Will return to TensorFlow computation graphs and explain
        
#%%
"""sample code to run full gradient descent:
"""

opt_operation = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as ss:
    # initialize variables in graph
    ss.run(tf.global_variables_initializer())
    # Gradient descent loop for 500 steps
    for _ in range(500):
        # select random minibatch
        indices = np.random.choice(n, batch_size)
        x_batch, y_batch = xx[indices], yy[indices]
        # do gradient descent step
        _, loss_val = ss.run([opt_operation, loss], feed_dict={x: x_batch, y: y_batch})

#%%


