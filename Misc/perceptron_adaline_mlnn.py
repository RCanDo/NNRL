# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 01:21:03 2019

https://sebastianraschka.com/faq/docs/diff-perceptron-adaline-neuralnet.html

@author: kasprark
"""

from mlxtend.evaluate import plot_decision_regions   #! ImportError: cannot import name 'plot_decision_regions'
from mlxtend.classifier import Perceptron
from mlxtend.classifier import Adaline
from mlxtend.classifier import MultiLayerPerceptron
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import matplotlib.gridspec as gridspec
import itertools

gs = gridspec.GridSpec(2, 2)#xw
X, y = make_moons(n_samples=100, random_state=123)
fig = plt.figure(figsize=(10,8))

ppn = Perceptron(epochs=50, eta=0.05, random_seed=0)
ppn.fit(X, y)
ada = Adaline(epochs=50, eta=0.05, random_seed=0)
ada.fit(X, y)

mlp = MultiLayerPerceptron(n_output=len(np.unique(y)),
                           n_features=X.shape[1],
                           n_hidden=150,
                           l2=0.0,
                           l1=0.0,
                           epochs=500,
                           eta=0.01,
                           alpha=0.0,
                           decrease_const=0.0,
                           minibatches=1,
                           shuffle_init=False,
                           shuffle_epoch=False,
                           random_seed=0)

mlp = mlp.fit(X, y)


for clf, lab, grd in zip([ppn, ppn, mlp],
                         ['Perceptron', 'Adaline', 'MLP (logistic sigmoid)'],
                         itertools.product([0, 1], repeat=2)):

    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf, legend=2)
    plt.title(lab)

plt.show()
