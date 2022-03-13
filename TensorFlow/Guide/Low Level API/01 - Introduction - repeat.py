#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: Introduction to low level API
part: 1 
subtitle:
version: 2.0
type: code chunks
keywords: [TensorFlow, low level API]
description: Examples; the same as v. 1.0 but without comments.
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
    date: 2019-01-29
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

import numpy as np
import tensorflow as tf

#%%

a = tf.constant(3., dtype=tf.float32)
b = tf.constant(4., )
total = a + b
print(a)
print(b)
print(total)

#%%

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())    ## run it every so often

#%%

sess = tf.Session()
sess.run(total)

#%%

norm = (a**2 + b**2)**(1/2)

sess.run(norm)
print(sess.run(norm))

sess.run({'ab':(a, b), 'tot_norm':(total, norm)})

#%%  

vec = tf.random_uniform(shape=(3,2))
vec1 = vec + 1
vec2 = vec + 2

print(vec)
print(vec1)
print(vec2)

sess.run(vec)
sess.run(vec)

sess.run((vec1, vec2))

#%%  A placeholder is a promise to provide a value later, like a function argument.
  
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

z = x + y

sess.run(z, feed_dict={x:1, y:2})
sess.run(z, feed_dict={x:[1, 3], y:[2, 4]})

writer.add_graph(tf.get_default_graph())

#%%  Datasets

my_data = [[0], [-2], [4], [9]]
slices = tf.data.Dataset.from_tensor_slices(my_data)

next_row = slices.make_one_shot_iterator().get_next()   # depricated

while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break

#%% STATEFUL operations must be initialized

r = tf.random_normal([10, 3])                             #! this is STATEFUL
dataset = tf.data.Dataset.from_tensor_slices(r)
iterator = dataset.make_initializable_iterator()
next_row = iterator.get_next()

sess.run(iterator.initializer)
while True:
    try:
        print(sess.run(next_row))
    except tf.errors.OutOfRangeError:
        break

#%% Layers
        
x = tf.placeholder(tf.float32, shape=[None, 3])
linear_model = tf.layers.Dense(units=1)

y = linear_model(x)    ## sth depricated...

writer.add_graph(tf.get_default_graph())

#%% layers must be initialized

init = tf.global_variables_initializer()

sess.run(init)

sess.run(y, {x:[[1, 2, 3], [4, 5, 6]]})
        
sess.run(y, {x: tf.random_normal([10, 3])})  
""" The value of a feed cannot be a tf.Tensor object. 
Acceptable feed values include Python scalars, strings, lists, numpy ndarrays, or TensorHandles
"""

sess.run(y, {x:  np.random.random([10, 3])})  

#%% re-run
sess.run(y, {x:[[1, 2, 3], [4, 5, 6]]})
sess.run(y, {x:[[1, 2, 3], [4, 5, 6]]})

sess.run(init)
sess.run(y, {x:[[1, 2, 3], [4, 5, 6]]})
sess.run(y, {x:[[1, 2, 3], [4, 5, 6]]})

#%%  Layer function shortcuts

x = tf.placeholder(tf.float32, shape=[None, 3])
y = tf.layers.dense(x, units=1)     # depricated - Use  keras.layers.dense() instead.

init = tf.global_variables_initializer()                #!  MUST BUE RUN AGAIN

sess.run(init)
sess.run(y, {x:[[1, 2, 3], [4, 5, 6]]})

writer.add_graph(tf.get_default_graph())

#%% Feature columns

features = { 'sales' : [[5], [6], [10], [7]],
             'department' : ['a', 'b', 'b', 'a']
           }

department_column = tf.feature_column.categorical_column_with_vocabulary_list( 
                     'department', ['a', 'b']
                    )

columns = [ tf.feature_column.numeric_column('sales'),
            tf.feature_column.indicator_column(department_column)
          ]


inputs = tf.feature_column.input_layer(features, columns)

#%%

var_init = tf.global_variables_initializer()
table_init = tf.tables_initializer()

sess.run((var_init, table_init))

sess.run(inputs)

writer.add_graph(tf.get_default_graph())

#%%  Training

x      = tf.constant([[1], [2], [3], [4]], dtype = tf.float32)
y_true = tf.constant([[0], [-1], [-2], [-3]], dtype = tf.float32)

lin_mod = tf.layers.Dense(units=1)
y_pred = lin_mod(x)

init = tf.global_variables_initializer()
sess.run(init)

sess.run(y_pred)


loss = tf.losses.mean_squared_error(y_true, y_pred)
sess.run(loss)

optimizer = tf.train.GradientDescentOptimizer(.01)
train = optimizer.minimize(loss)

for i in range(100):
    _, ls = sess.run((train, loss))
    print(ls)
   
    
#%% 
    
    
    
