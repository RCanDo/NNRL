#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: The Linear Regression Algorithm
part: 1
subtitle:
version: 1.0
type: code chunks
keywords: [TensorFlow, Machine Learning, regression]
description: examples 
sources:
    - title: Getting Started with TensorFlow - Giancarlo Zaccone, 2016 
      chapter: 3. Starting with Machine Learning
      pages: 67-
      link: "D:/bib/Python/TensorFlow/Getting Started With TensorFlow (178).pdf"
      date: 2016
      authors: 
          - fullname: Giancarlo Zaccone
      usage: code chunks copy and other examples
file:
    usage: 
        interactive: True
        terminal: False
    name: "01 - regression.py"
    path: "D:/ROBOCZY/Python/TensorFlow/Zaccone - Getting Started With TF/03 - ML/"
    date: 2019-09-19
    authors:   
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: 
              - arkadiusz.kasprzyk@tieto.com
              - akasp666@google.com
              - arek@staart.pl      
"""              

#%%
"""
cd "D:\ROBOCZY\Python\TensorFlow\Zaccone - Getting Started With TF\03 - ML"
pwd
ls
%reset
"""
#%%
import os, glob
os.chdir("D:\\ROBOCZY\\Python\\TensorFlow\\Zaccone - Getting Started With TF\\03 - ML")    
"""
Anaconda Prompt
>cd <path_to_the_working_dir>
>del event*
or
"""
for file in glob.glob(".\\events*"): 
    os.remove(file)

#%%
import numpy as np
import matplotlib.pyplot as plt

#import keras.backend as K
import tensorflow as tf

#%%
"""
the two important concepts of machine learning: 
    - the cost function,
    - gradient descent algorithms.
"""

N = 500
a, b = .22, .78
sigma = .1      # scale == std.dev.

xx = np.random.normal(0, .5, N)
yy = a * xx + b + np.random.normal(0, .1, N)

plt.plot(xx, yy, '.', label='Input Data')
plt.legend()
plt.show()

#%%

# initial values of parameters
a_hat = tf.Variable(tf.random_uniform([1], -1., 1.))
b_hat = tf.Variable(tf.zeros([1]))
yy_hat = a_hat * xx + b_hat

cost_function = tf.reduce_mean(tf.square(yy - yy_hat))   # Equivalent to np.mean(); Computes the mean of elements across dimensions of a tensor.
optimizer = tf.train.GradientDescentOptimizer(.5)
train = optimizer.minimize(cost_function)

model = tf.global_variables_initializer()  #! tf.initialize_all_variables() is depricated!

with tf.Session() as ss:
    ss.run(model)
    for k in range(21):
        ss.run(train)
        if k % 5 == 0:
            plt.plot(xx, yy, '.', label = 'step {}'.format(k))
            yy_hat = ss.run(a_hat) * xx + ss.run(b_hat)
            plt.plot(xx, yy_hat)
            plt.legend()
            plt.show()
            
#%%

writer = tf.summary.FileWriter(".")
writer.add_graph(tf.get_default_graph())  # after every modification of a graph

"""
> py <file.py e.g. this file>
> tensorboard --logdir ./ --host 127.0.0.1
in browser open:  http://127.0.0.1:6006/
"""

#%%
            
