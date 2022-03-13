# -*- coding: utf-8 -*-
"""
title: Ramsudar - TensorFlow Tutorial
link: https://www.tensorflow.org/get_started/eager
date: Created on Wed Jun  7 15:17:43 2018
author: kasprark
"""


#%%

import numpy as np

a = np.zeros((2, 2))
a
b = np.ones((2, 2))
b


np.sum(b, axis=1)   # array([2., 2.])
a.shape             # (2, 2)
np.reshape(a, (1, 4))   # array([[ 0., 0., 0., 0.]])
a

#%%

import tensorflow as tf

