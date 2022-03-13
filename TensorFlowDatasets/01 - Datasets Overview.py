#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: TensorFlow Datasets Overview
part: 1 
subtitle:
version: 1.0
type: code chunks
keywords: [TensorFlow Datasets, MNIST, loading data, importing data]
description: examples 
sources:
    - title: TensorFlow Datasets
      link: https://www.tensorflow.org/datasets/overview
      usage: copy
    - link: https://www.tensorflow.org/guide/datasets
file:
    usage: 
        interactive: True
        terminal: False
    name: "01 - Datasets Overview.py"
    path: "D:/ROBOCZY/Python/TensorFlow/Datasets/"
    date: 2019-09-23
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
import os
os.chdir("D:\\ROBOCZY\\Python\\TensorFlow\\Zaccone - Getting Started With TF\\03 - ML")    

#%%
    
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

#%%

tf.enable_eager_execution()

tfds.list_builders()

#%%

mnist_train = tfds.load(name="mnist", split="train")

assert isinstance(mnist_train, tf.data.Dataset)
mnist_train

#%% with versioning

mnist = tfds.load("mnist:1.*.*")
mnist  # dict
mnist['test']
mnist['train']

#%% DatasetBuilder
"""
tfds.load() is really a thin conveninence wrapper around DatasetBuilder. 
We can accomplish the same as above directly with the MNIST DatasetBuilder.
"""

mnist_builder = tfds.builder("mnist")
mnist_builder.download_and_prepare()
mnist_train = mnist_builder.as_dataset(split="train")
mnist_train

#%% Feature dictionaries

for mnist_example in mnist_train.take(1):  # Only take a single example
    image, label = mnist_example["image"], mnist_example["label"]
    plt.imshow(image.numpy()[:, :, 0].astype(np.float32), cmap=plt.get_cmap("gray"))
    print("Label: %d" % label.numpy())


#%% Input pipelines
"""
Once you have a tf.data.Dataset object, it's simple to define the rest of an input pipeline 
suitable for model training by using the tf.data API.
Here we'll repeat the dataset so that we have an infinite stream of examples, shuffle, 
and create batches of 32.
"""

mnist_train = mnist_train.repeat().shuffle(1024).batch(32)

# `prefetch` will enable the input pipeline to asynchronously fetch _batches_ while
# your model is training.
mnist_train = mnist_train.prefetch(tf.data.experimental.AUTOTUNE)

# Now you could loop over batches of the dataset and train for batch in mnist_train: ...

#%% DatasetInfo
"""
After generation, the builder contains useful information on the dataset:
"""
info = mnist_builder.info
print(info)

print(info.features)
print(info.features["label"].num_classes)
print(info.features["label"].names)

#%% You can also load the DatasetInfo directly with tfds.load() using with_info=True.

mnist_test, info = tfds.load("mnist", split="test", with_info=True)
info
mnist_test



#%%