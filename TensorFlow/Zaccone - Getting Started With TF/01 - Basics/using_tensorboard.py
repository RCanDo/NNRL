# -*- coding: utf-8 -*-
# using_tensorboard.py 
"""
title: 
chapter: 1. Basic Concepts / How To Use TensorBoard, p.28
book: Getting Started with TensorFlow - Giancarlo Zaccone, 2016
link: "D:\bib\Python\TensorFlow\Getting Started With TensorFlow (178).pdf"
date: 2019-02-05 Tue 14:24:26
author: kasprark
"""

#%%

# cd "D:\ROBOCZY\Python\TensorFlow\Zaccone - Getting Started With TF\01 - Basics"
# pwd

#%%

import os
import glob
import tensorflow as tf

#%%

def main():
    """
    Anaconda Prompt
    >cd <path_to_the_working_dir>
    """
    
    os.chdir("D:\\ROBOCZY\\Python\\TensorFlow\\Zaccone - Getting Started With TF\\01 - Basics")    
    for file in glob.glob(".\\events*"): 
        os.remove(file)

    """ or from Anaconda Prompt
    remove all the files  `event*`  from current dir
    >del event*
    """
    
    #--- do something ------------------------
    a = tf.constant(10, name='a')
    b = tf.constant(90, name='b')
    y = tf.Variable(a + b*2, name="y")
    
    model = tf.global_variables_initializer()
    
    with tf.Session() as ss:
        merged = tf.summary.merge_all()
        
        ss.run(model)
        print(ss.run(y))
    #-----------------------------------------


    writer = tf.summary.FileWriter(".")
    writer.add_graph(tf.get_default_graph())  # after every modification of a graph
    
    """
    > py <file.py e.g. this file>
    > tensorboard --logdir ./ --host 127.0.0.1
    in browser open:  http://127.0.0.1:6006/
    """


#%%
    
if __name__ == "__main__":
    main()    
    
