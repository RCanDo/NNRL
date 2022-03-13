# -*- coding: utf-8 -*-
"""
file: 02-using_gpu.py 
title: 
chapter: 
book: 
link: https://www.tensorflow.org/guide/using_gpu
date: 2019-05-17 Fri 19:42:16
author: kasprark
email: arkadiusz.kasprzyk@tieto.com
"""

import numpy as np
import tensorflow as tf

#%%  Supported devices
"""
On a typical system, there are multiple computing devices. 
In TensorFlow, the supported device types are CPU and GPU. 
They are represented as strings. For example:

    "/cpu:0": The CPU of your machine.
    "/device:GPU:0": The GPU of your machine, if you have one.
    "/device:GPU:1": The second GPU of your machine, etc.

If a TensorFlow operation has both CPU and GPU implementations, 
the GPU devices will be given priority when the operation is assigned to a device. 
For example, matmul has both CPU and GPU kernels. 
On a system with devices cpu:0 and gpu:0, gpu:0 will be selected to run matmul.
"""

#%% Logging Device placement
"""
To find out which devices your operations and tensors are assigned to, 
create the session with `log_device_placement` configuration option set to True.
"""

#'''
# create a graph
a = tf.constant([1., 2., 3., 4., 5., 6.], shape=[2, 3], name= 'a')
b = tf.constant([1., 2., 3., 4., 5., 6.], shape=[3, 2], name= 'b')
c = tf.matmul(a, b)

# creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# runs the op.
#print(sess.run(c))
#'''

#%% Manual device placement

#'''
# Creates a graph.
with tf.device('/cpu:0'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
#print(sess.run(c))
#'''

#%% Allowing GPU memory growth
"""
By default, TensorFlow maps nearly all of the GPU memory of all GPUs 
(subject to CUDA_VISIBLE_DEVICES) visible to the process. 
This is done to more efficiently use the relatively precious GPU memory resources on the devices 
by reducing memory fragmentation.

In some cases it is desirable for the process to only allocate a subset of the available memory, 
or to only grow the memory usage as is needed by the process. 
TensorFlow provides two Config options on the Session to control this.

The first is the `allow_growth` option, which attempts to allocate only as much GPU memory 
based on runtime allocations: it starts out allocating very little memory, 
and as Sessions get run and more GPU memory is needed, 
we extend the GPU memory region needed by the TensorFlow process. 
Note that we do not release memory, since that can lead to even worse memory fragmentation. 
To turn this option on, set the option in the ConfigProto by:
"""
#'''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True        #!
session = tf.Session(config=config, ...)
#'''
"""
The second method is the per_process_gpu_memory_fraction option, which determines the fraction 
of the overall amount of memory that each visible GPU should be allocated. 
For example, you can tell TensorFlow to only allocate 40% of the total memory of each GPU by:
"""
#'''
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config, ...)
#'''
"""
This is useful if you want to truly bound the amount of GPU memory available 
to the TensorFlow process.
"""

#%% Using a single GPU on a multi-GPU system
"""
If you have more than one GPU in your system, 
the GPU with the lowest ID will be selected by default. 
If you would like to run on a different GPU, 
you will need to specify the preference explicitly:
"""
#'''
# Creates a graph.
with tf.device('/device:GPU:2'):
  a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
  b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
  c = tf.matmul(a, b)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(c))
#'''
"""
If the device you have specified does not exist, you will get `InvalidArgumentError`
"""

#%% Using multiple GPUs
"""
If you would like to run TensorFlow on multiple GPUs, 
you can construct your model in a `multi-tower` fashion 
where each `tower` is assigned to a different GPU. For example:
"""
#'''
# Creates a graph.
c = []
for d in ['/device:GPU:2', '/device:GPU:3']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))
with tf.device('/cpu:0'):
  sum = tf.add_n(c)
# Creates a session with log_device_placement set to True.
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# Runs the op.
print(sess.run(sum))
#'''
"""
"""

#%%
if __name__ == "__main__":
    print(sess.run(c))