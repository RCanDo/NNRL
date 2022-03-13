# -*- coding: utf-8 -*-
"""
title: Neural Networks For Beginners
    book: Thomas - An Introduction To Neural Networks For Beginners
        file: "Thomas - An Introduction To Neural Networks For Beginners (Python) (40).pdf"
    chapter: 1. Introduction To Neural Networks   
author: Arkadiusz Kasprzyk
email: akasp@interia.pl; arkadiusz.kasprzyk@tieto.com
date: Mon May 28 20:34:42 2018
"""

import matplotlib.pylab as plt
import numpy as np

#%%
"""
"""

x = np.arange(-8, 8, .1)
f = 1 / (1 + np.exp(-x))
plt.plot(x, f)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

#%% weights

w1 = .5
w2 = 1.
w3 = 2.

l1 = 'w = 0.5'
l2 = 'w = 1.0'
l3 = 'w = 2.0'

for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
    f = 1 / (1 + np.exp(-x * w))
    plt.plot(x, f, label=l)
    
plt.xlabel('x')
plt.ylabel('h_w(x)')
plt.legend(loc = 2)
plt.show()

#%% bias

w = 5.

b1 = -8.
b2 = 0.
b3 = 8.

l1 = 'b = -8.0'
l2 = 'b = 0.0'
l3 = 'b = 8.0'

for b, l in [(b1, l1), (b2, l2), (b3, l3)]:
    f = 1 / (1 + np.exp(-(x * w + b)))
    plt.plot(x, f, label=l)
    
plt.xlabel('x')
plt.ylabel('h_w(x)')
plt.legend(loc = 2)
plt.show()

#%%

