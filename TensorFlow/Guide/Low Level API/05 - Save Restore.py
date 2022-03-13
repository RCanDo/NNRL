# -*- coding: utf-8 -*-
"""
title: Variables
link: https://www.tensorflow.org/guide/saved_model
date: 2019-02-01 Fri 16:06:25
author: kasprark
"""

#%%

cd D:\ROBOCZY\Python\TensorFlow\Guide\LowLevelAPI\
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
>cd ROBOCZY\Python\TensorFlow\Guide\LowLevelAPI\Introduction

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


#%%  ...
    
# Create some variables.
v1 = tf.get_variable("v1", shape=[3], initializer = tf.zeros_initializer)
v2 = tf.get_variable("v2", shape=[5], initializer = tf.zeros_initializer)

inc_v1 = v1.assign(v1+1)
dec_v2 = v2.assign(v2-1)

# Add an op to initialize the variables.
init_op = tf.global_variables_initializer()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()


#%%

# Later, launch the model, initialize the variables, do some work, and save the
# variables to disk.
with tf.Session() as sess:
  sess.run(init_op)
  # Do some work with the model.
  inc_v1.op.run()
  dec_v2.op.run()
  # Save the variables to disk.
  save_path = saver.save(sess, "/tmp/model.ckpt")
  print("Model saved in path: %s" % save_path)
  
  
#%%
  



