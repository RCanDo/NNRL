#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: Variables
part: 1 
subtitle:
version: 2.0
type: code chunks
keywords: [TensorFlow, low level API]
description: Examples.
sources:
    - title: Variables
      link: https://www.tensorflow.org/guide/variables
      usage: copy
file:
    usage: 
        interactive: True
        terminal: False
    name: "02 - Variables.py"
    path: "D:/ROBOCZY/Python/TensorFlow/Guide/Low Level API/"
    date: 2018-08-25
    authors:   
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: 
              - arkadiusz.kasprzyk@tieto.com
              - akasp666@google.com
              - arek@staart.pl      
"""  
#%%
#%%
"""
TensorFlow variables

A TensorFlow variable is the best way to represent _shared_, _persistent_ state 
manipulated by your program.

Variables are manipulated via the tf.Variable class. 
A tf.Variable represents a tensor whose value can be _changed_ by running ops on it. 
Specific ops allow you to read and modify the values of this tensor. 
Higher level libraries like tf.keras use tf.Variable to store model parameters. 
This guide covers how to create, update, and manage tf.Variables in TensorFlow.
"""

#%%
"
cd "D:\ROBOCZY\Python\TensorFlow\Guide\Low Level API\"
pwd

#%%
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#%%
"""
Anaconda Prompt
>d:
>cd "ROBOCZY\Python\TensorFlow\Guide\Low Level API\"
remove all the files  `event*`  from current dir
>del event*
"""

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

"""
> tensorboard --logdir ./ --host 127.0.0.1
in browser open:  http://127.0.0.1:6006/
"""

#%%

def run_and_show(lst):
    init = tf.global_variables_initializer()
    with tf.Session() as ss:
        ss.run(init)
        for tensor in lst:
            print(ss.run(tensor))
    writer.add_graph(tf.get_default_graph())

#%%  Variables 
"""
Create a variable
To create a variable, simply provide the initial value
"""
my_variable = tf.Variable(tf.zeros([1., 2., 3.]))
my_variable
run_and_show([my_variable])    
"""
This creates a variable which is a three-dimensional tensor with shape [1, 2, 3] filled with zeros. 
This variable will, by default, have the dtype tf.float32. 
The dtype is, if not specified, inferred from the initial value.
"""
#%%
"""
If there's a tf.device scope active, the variable will be placed on that device; 
otherwise the variable will be placed on the "fastest" device compatible with its dtype 
(this means most variables are automatically placed on a GPU if one is available). 
For example, the following snippet creates a variable named v 
and places it on the second GPU device:
"""
with tf.device("/device:GPU:1"):  # !!! DO NOT RUN !!!
  v = tf.Variable(tf.zeros([10, 10]))
"""
Ideally though you should use the tf.distribute API, 
as that allows you to write your code once and have it work under many different distributed setups.
"""  
#%% old way -- depricated!!! 
# creating variable
my_var = tf.get_variable("my_var", shape=[1, 2, 3])
my_var

run_and_show([my_var])

my_var_int = tf.get_variable("my_var_int", [1, 2, 3], dtype=tf.int32,
                                  initializer=tf.zeros_initializer)
my_var_int

run_and_show([my_var_int])

# see more on the "old way" in "03 - Variables - old.py"

#%%
"""
Use a variable

To use the value of a tf.Variable in a TensorFlow graph, simply treat it like a normal tf.Tensor:
"""
v = tf.Variable(0.0)
w = v + 1  # w is a tf.Tensor which is computed based on the value of v.
           # Any time a variable is used in an expression it gets automatically
           # converted to a tf.Tensor representing its value.
w           
run_and_show([v, w])           
"""
To assign a value to a variable, use the methods assign(), assign_add(), and friends 
in the tf.Variable class. 
For example, here is how you can call these methods:
"""
v = tf.Variable(0.0)
v
run_and_show([v])

v.assign_add(1)
run_and_show([v])  # still  0 ...

u = v.assign_add(1)
u
run_and_show([v, u])

"""
Most TensorFlow optimizers have specialized ops that efficiently update the values of variables 
according to some gradient descent-like algorithm. 
See tf.keras.optimizers.Optimizer for an explanation of how to use optimizers.

You can also explicitly read the current value of a variable, using read_value:
"""
v = tf.Variable(0.0)
v.assign_add(1)
v.read_value()  # 1.0
# nothing... ??? just another tensor...

z = v.read_value() 
z
run_and_show([v, z])  # 0 0  ... sth wrong; maybe tf.VERSION wrong?

"""
When the last reference to a tf.Variable goes out of scope its memory is freed.
"""

#%%
"""
Keep track of variables

A Variable in TensorFlow is a Python object. 
As you build your layers, models, optimizers, and other related tools, 
you will likely want to get a list of all variables in a (say) model.

A common use case is implementing Layer subclasses. 
The Layer class recursively tracks variables set as instance attributes:
"""
class MyLayer(tf.keras.layers.Layer):

  def __init__(self):
    super(MyLayer, self).__init__()
    self.my_var = tf.Variable(1.0)
    self.my_var_list = [tf.Variable(x) for x in range(10)]

class MyOtherLayer(tf.keras.layers.Layer):

  def __init__(self):
    super(MyOtherLayer, self).__init__()
    self.sublayer = MyLayer()
    self.my_other_var = tf.Variable(10.0)

m = MyOtherLayer()
print(len(m.variables))  # 12 (11 from MyLayer, plus my_other_var)

"""
If you aren't developing a new Layer, TensorFlow also features a more generic tf.Module base class 
which only implements variable tracking. 
Instances of tf.Module have a variables and a trainable_variables property 
which return all (trainable) variables reachable from that model, 
potentially navigating through other modules (much like the tracking done by the Layer class).
"""
