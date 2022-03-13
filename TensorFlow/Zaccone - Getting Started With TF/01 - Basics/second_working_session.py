# -*- coding: utf-8 -*-
# second_working_session.py 
"""
title: 
chapter: 1. Basic Concepts / Tensor Flow Programming Model, p.27
book: Getting Started with TensorFlow - Giancarlo Zaccone, 2016
link: "D:\bib\Python\TensorFlow\Getting Started With TensorFlow (178).pdf"
date: 2019-02-05 Tue 13:07:26
author: kasprark
"""

#%%

# cd "D:\ROBOCZY\Python\TensorFlow\Zaccone - Getting Started With TF\01 - Basics"
# pwd

#%%
import os
import tensorflow as tf

#%%

def main():
    
    os.chdir("D:\\ROBOCZY\\Python\\TensorFlow\\Zaccone - Getting Started With TF\\01 - Basics")    

    writer = tf.summary.FileWriter('.')

    a = tf.placeholder("int32")
    b = tf.placeholder("int32")
    
    y = tf.multiply(a, b)
    
    ss = tf.Session()
    
    print(ss.run(y, feed_dict={a: 2, b: 5}))

    writer.add_graph(tf.get_default_graph())
    """
    > tensorboard --logdir ./ --host 127.0.0.1
    in browser open:  http://127.0.0.1:6006/
    """


if __name__ == "__main__":
    main()    
