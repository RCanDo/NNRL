#! python3
# -*- coding: utf-8 -*-
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Keras - Intro For Engineers
subtitle:
version: 1.0
type: tutorial
keywords: [tensor, Keras, Tensor Flow]
description: |
    Everything you need to know to use Keras to build real-world machine learning solutions.
remarks:
    - etc.
todo:
    - problem 1
sources:
    - title: Introduction to Keras for Engineers
      chapter:
      pages:
      link: https://keras.io/getting_started/intro_to_keras_for_engineers/
      date: 2020/04/28
      authors:
          - nick: fchollet
            fullname:
            email: https://twitter.com/fchollet
      usage: |
file:
    usage:
        interactive: True   # if the file is intended to be run interactively e.g. in Spyder
        terminal: False     # if the file is intended to be run in a terminal
    name: 01-intro_for_engineers.py
    path: E:/ROBOCZY/Python/Keras/Intro For Engineers/
    date: 2021-09-13
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
"""

#%% This is block delimiter very useful for interactive work like e.g. in Spyder (part of Anaconda)

#%%
from rcando.ak.builtin import * #flatten, paste
from rcando.ak.nppd import * #data_frame
import os, sys, json

ROOTS = json.load(open('roots.json'))
WD = os.path.join(ROOTS['Works'], "Python/Pandas/User Guide/")   #!!! adjust
os.chdir(WD)

print(os.getcwd())


#%% Block delimiters allows to run separated blocks of code by one key-stroke
# e.g. Shift+Enter in Spyder


#%%
