# -*- coding: utf-8 -*-
# first_working_session.py 
"""
title: 
chapter: 1. Basic Concepts / First Working Session, p.23
book: Getting Started with TensorFlow - Giancarlo Zaccone, 2016
link: "D:\bib\Python\TensorFlow\Getting Started With TensorFlow (178).pdf"
date: 2019-02-05 Tue 12:36:42
author: kasprark
"""

#%%

#cd "D:\ROBOCZY\Python\TensorFlow\Zaccone - Getting Started With TF\01 - Basics"
#pwd

#%%
import tensorflow as tf

#%%

def main():
    x = tf.constant(1, name='x')
    y = tf.Variable(x+9, name='y')
    print(y)
    
    ## model = tf.initialize_all_variables()    # depricated
    model = tf.global_variables_initializer()
    
    with tf.Session() as ss:
        ss.run(model)
        print(ss.run(y))


if __name__ == "__main__":
    main()