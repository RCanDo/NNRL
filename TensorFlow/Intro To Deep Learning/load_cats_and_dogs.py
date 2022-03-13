#! python3
"""
---  
title: Data Loader
subtitle: 
version: 1.0
type: function
keywords: [data loader]
description: |
    Data loader for the data from the course in source (below).
remarks: 
todo:
sources: 
    - title: Deep Learning basics with Python, TensorFlow and Keras
      chapter: p.2 - p.6
      link: https://pythonprogramming.net/tensorboard-analysis-deep-learning-python-tensorflow-keras/
file:
    usage:
        interactive: True
        terminal: False
    name: load_cats_and_dogs.py
    path: ./Python/TensorFlow/Intro To Deep Learning/
    date: 2019-06-26 
    authors: 
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: arkadiusz.kasprzyk@tieto.com
    project: AIML Competence Development / Gym Training             
    company: Tieto
"""

import pickle
import random
import numpy as np

def load_cats_and_dogs(train=1024, validation=256, sample=True):
    """
    train      int   > 0
    validation int   > 0
    sample     bool  True
    """

    pickle_in = open("X.pickle","rb")
    X = pickle.load(pickle_in)
    
    pickle_in = open("y.pickle","rb")
    y = pickle.load(pickle_in)
    
    pickle_in = open("X_test.pickle","rb")
    X_val = pickle.load(pickle_in)
    
    pickle_in = open("y_test.pickle","rb")
    y_val = pickle.load(pickle_in)
    
    if train == -1:
        train = len(y) 
    if validation == -1:
        validation = len(y_val) 

    if sample:    
        if train < len(y):
            X, y = zip(*random.sample(list(zip(X, y)), train))
            X = np.array(X)
            y = np.array(y)
        
        if validation < len(y_val):
            X_val, y_val = zip(*random.sample(list(zip(X_val, y_val)), validation))
            X_val = np.array(X_val)
            y_val = np.array(y_val)
    
    else:
        X = X[:train]
        y = y[:train]
        X_val = X_val[:validation]
        y_val = y_val[:validation]
    
    
    X = X/255.0
    X_val = X_val/255.0
    
    return X, y, X_val, y_val