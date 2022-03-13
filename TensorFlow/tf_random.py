# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 20:29:25 2019

@author: kasprark
"""

import tensorflow as tf
import numpy as np

samples = tf.random.categorical(tf.log([[10., 10.]]), 5)
samples

ss = tf.Session()
ss.run(samples)

samples = tf.random.categorical(tf.log([[1., 1., 1., 1.]]), 5)
ss.run(samples[0])

samples = tf.random.categorical(tf.log([[1., 1., 1., 1.]]), 10)
ss.run(samples)

samples = tf.random.categorical(tf.log([[2., 1.]]), 100)
ss.run(samples)

samples = tf.random.categorical(tf.log([[2., 1.]]), 100)
ss.run(tf.reduce_sum(samples))

samples = tf.random.categorical(tf.log([[2., 1.]]), 100)
ss.run(tf.reduce_sum(samples))

ss.run(tf.summary.histogram("a", samples))  # NO!!!
# so how to make table of values???

samples = tf.random.categorical(tf.log([[1., 1.], [1., 1.]]), 5)
ss.run(samples)


samples = tf.random.categorical(tf.log([[1.] * 3]), 5)
ss.run(samples)

samples = tf.random.categorical(tf.log([[1.] * 3]), 1)
ss.run(samples[0][0])

tf.Variable()


