#! python3
"""
---  
title: Loading in your own data
subtitle: 
version: 1.0
type: example
keywords: [callback, TensorBorad, learning statistics]
description: |
    ...
remarks: 
todo:
sources: 
    - title: Deep Learning basics with Python, TensorFlow and Keras
      chapter: p.2 - Loading in your own data
      link: https://pythonprogramming.net/tensorboard-analysis-deep-learning-python-tensorflow-keras/
      usage: | 
          Copy of the code to see how it works.
file:
    usage:
        interactive: True
        terminal: False
    name: "02 - Loading In Your Own Data.py"
    path: ./Python/TensorFlow/Intro To Deep Learning/
    date: 2019-06-19 
    authors: 
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: arkadiusz.kasprzyk@tieto.com
    project:         
    company: Tieto
"""  

#%%

cd "D:/ROBOCZY/Python/TensorFlow/Intro To Deep Learning/"
ls

#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm


DATADIR = os.path.join(os.getcwd(), "kagglecatsanddogs_3367a\\PetImages")

CATEGORIES = ["Dog", "Cat"]

#%%

for category in CATEGORIES:  # do dogs and cats
    path = os.path.join(DATADIR, category)  # create path to dogs and cats
    for img in os.listdir(path)[:3]:  # iterate over each image per dogs and cats
        img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
        # plot
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(img_array, cmap='gray')  # graph it
        #plt.show()  # display!
        #break  # we just want one for now so break
    #break  #...and one more!
    
#%%
    
img_array
img_array.shape    

IMG_SIZE = 100

new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
plt.imshow(new_array, cmap='gray')
#plt.show()

#%%

def dataset():
    data = []
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR, category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat
    
        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))
    return data

data = dataset()

#%%

type(data)
len(data)
data[0]
data[0][0]
data[0][0][0]
data[0][0][0, 1]
data[0][0][:, 1]
plt.imshow(data[0][0], cmap='gray')
data[0][1]

#%%

import random

random.shuffle(data)

for img, cat in data[:100]: print(cat)

n = len(data)
n0 = round(n*.9)
n
n0
n-n0

data_train = data[:n0]
data_test  = data[n0:]

#%%

X = []
y = []

for img, cat in data_train:
    X.append(img)
    y.append(cat)
    

X_test = []
y_test = []

for img, cat in data_test:
    X_test.append(img)
    y_test.append(cat)
    
#%%
# we need to reshape to hava a proper input into NN
    
print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
print(np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)[:2])

X      = np.array(X     ).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
X_test = np.array(X_test).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
# y does not need it

#%% saving  data

import pickle

pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_out = open("X_test.pickle","wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()


#%% loading by hand

pickle_in = open("X.pickle","rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle","rb")
y = pickle.load(pickle_in)

pickle_in = open("X_test.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)



#%% sampling from data

X = X[:5]
y = y[:5]

X_, y_ = zip(*random.sample(list(zip(X, y)), 3))     #!#!#! a smart way of sampling
X_ = np.array(X_)
y_ = np.array(y_)


# check if it's all right

X[0][10][10], X[1][10][10], X[2][10][10], X[3][10][10], X[4][10][10]
y

X_[0][10][10], X_[1][10][10], X_[2][10][10]
y_

#%%
"""!!!
the loading data procedure is written in `load_cats_and_dogs.py`
"""