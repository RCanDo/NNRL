#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: TensorFlow Importing Data Guide
part: 1 
subtitle:
version: 1.0
type: code chunks
keywords: [TensorFlow Datasets, MNIST, loading data, importing data]
description: examples 
sources:
    - title: TensorFlow Importing Data Guide
      link: https://www.tensorflow.org/guide/datasets
      usage: copy
    - link: https://www.tensorflow.org/datasets/overview
      usage: ...
file:
    usage: 
        interactive: True
        terminal: False
    name: "02 - Importing Data Guide.py"
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

import numpy as np
import tensorflow as tf

# tf.enable_eager_execution()

ss = tf.Session()

#%%

ds1 = tf.data.Dataset.from_tensor_slices(tf.random_uniform([4, 10]))

ds1.output_types   # ==> "tf.float32"       # depricated
tf.compat.v1.data.get_output_types(ds1)

ds1.output_shapes  # ==> "(10,)"           # depricated
tf.compat.v1.data.get_output_shapes(ds1)

#%%

ds2 = tf.data.Dataset.from_tensor_slices(
   (tf.random_uniform([4]),
    tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)))

print(ds2.output_types)   # ==> "(tf.float32, tf.int32)"
print(ds2.output_shapes)  # ==> "((), (100,))"

#%%

ds3 = tf.data.Dataset.zip((ds1, ds2))
print(ds3.output_types)  # ==> (tf.float32, (tf.float32, tf.int32))
print(ds3.output_shapes)  # ==> "(10, ((), (100,)))"

#%%

ds4 = tf.data.Dataset.from_tensor_slices(
   {"a": tf.random_uniform([4]),
    "b": tf.random_uniform([4, 100], maxval=100, dtype=tf.int32)})
    
print(ds4.output_types)  # ==> "{'a': tf.float32, 'b': tf.int32}"
print(ds4.output_shapes)  # ==> "{'a': (), 'b': (100,)}"

#%%
"""
The Dataset transformations support datasets of any structure. 
When using the Dataset.map(), Dataset.flat_map(), and Dataset.filter() transformations, 
which apply a function to each element, 
the element structure determines the arguments of the function:
"""
dataset1 = dataset1.map(lambda x: ...)

dataset2 = dataset2.flat_map(lambda x, y: ...)

# Note: Argument destructuring is not available in Python 3.
dataset3 = dataset3.filter(lambda x, (y, z): ...)

#%% map()  https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map
ds5 = tf.data.Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
ds5.map(lambda x: x + 1)           # ==> [ 2, 3, 4, 5, 6 ]
# ...

#%% flat_map()  https://www.tensorflow.org/api_docs/python/tf/data/Dataset#flat_map
"""
Use flat_map if you want to make sure that the order of your dataset stays the same. 
For example, to flatten a dataset of batches into a dataset of their elements:
"""
ds6 = tf.data.Dataset.from_tensor_slices([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])
ds6.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x + 1)) # ==> #  [ 2, 3, 4, 5, 6, 7, 8, 9, 10 ]

#%%
#%% Creating an iterator

"""
Once you have built a Dataset to represent your input data, the next step is to create an Iterator 
to access elements from that dataset. 
The `tf.data` API currently supports the following iterators, in increasing level of sophistication:
    one-shot,
    initializable,
    reinitializable, and
    feedable.
"""
#%% one-shot iterator
"""
A one-shot iterator is the simplest form of iterator, 
which only supports iterating once through a dataset, with no need for explicit initialization. 
One-shot iterators handle almost all of the cases 
that the existing queue-based input pipelines support, 
but they do not support parameterization. 
Using the example of Dataset.range():
"""
ds7 = tf.data.Dataset.range(100)
iterator = ds7.make_one_shot_iterator()   # depricated
next_element = iterator.get_next()

# you may run it only once! after run one must reinitialize iterator
for i in range(100):    # hence I need to know the nr of elements in dataset... no good...
  value = ss.run(next_element)
  assert i == value
  # print(i) # OK
  print(value) # OK
  
# infinite loop...
for d in ds7:
    print(d)

"""
!!! Note: Currently, one-shot iterators are the only type that is easily usable with an Estimator.
"""
#%% examples from above

itr1 = ds1.make_one_shot_iterator()
"""!!! ValueError: Failed to create a one-shot iterator for a dataset. 
`Dataset.make_one_shot_iterator()` does not support datasets that capture _stateful_ objects, 
such as a `Variable` or `LookupTable`. 
In these cases, use `Dataset.make_initializable_iterator()`. 
(Original error: Cannot capture a stateful node 
(name:random_uniform_1/RandomUniform, type:RandomUniform) by value.)
"""
...


#%% try it with .map() and .flat_map() as in examples above
ds8 = tf.data.Dataset.range(1, 6)  # ==> [1, 2, 3, 4, 5]
itr8 = ds8.make_one_shot_iterator()
nextel8 = itr8.get_next()

for i in range(1, 6):
    print(ss.run(nextel8))

#%%
ds8map = ds8.map(lambda x: x + 1)   
itr8map = ds8map.make_one_shot_iterator()
nextel8map = itr8map.get_next()

for i in range(1, 6):
    print(ss.run(nextel8map))

#%%
# or on the fly
nextel8map2 = ds8.map(lambda x: x + 2).make_one_shot_iterator().get_next()
for i in range(1, 6):
    print(ss.run(nextel8map2))

#%% let's see wat is the difference between .map() and .flat_map()
    
ds9 = tf.data.Dataset.from_tensor_slices([ [1, 2, 3], [4, 5, 6], [7, 8, 9] ])

ds9map = ds9.flat_map(lambda x: x + 1)  # TypeError: `map_func` must return a `Dataset` object. 
                                        # Got <class 'tensorflow.python.data.util.structure.TensorStructure'>
ds9map = ds9.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x + 1)) # ==> #  [ 2, 3, 4, 5, 6, 7, 8, 9, 10 ]
itr9map = ds9map.make_one_shot_iterator()
nextel9map = itr9map.get_next()

for i in range(9):
    print(ss.run(nextel9map))
    
# complicated ...    
    

#%%  initializable iterator 
"""
An initializable iterator requires you to run an explicit 
`iterator.initializer` operation before using it. 
In exchange for this inconvenience, it enables you to parameterize the definition of the dataset, 
using one or more tf.placeholder() tensors that can be fed when you initialize the iterator. 
Continuing the Dataset.range() example:
"""

max_value = tf.placeholder(tf.int64, shape=[])
ds10 = tf.data.Dataset.range(max_value)
itr10 = ds10.make_initializable_iterator()    # depricated
nextel10 = itr10.get_next()

# Initialize an iterator over a dataset with 10 elements.
ss.run(itr10.initializer, feed_dict={max_value: 10})

for i in range(10):             # how to get max_value from tf.
  value = ss.run(nextel10)
  assert i == value

# Initialize the same iterator over a dataset with 100 elements.
ss.run(itr10.initializer, feed_dict={max_value: 100})
for i in range(100):
  value = ss.run(nextel10)
  assert i == value

#%%
"""
A reinitializable iterator can be initialized from multiple different Dataset objects. 
For example, you might have a training input pipeline that uses random perturbations 
to the input images to improve generalization, 
and a validation input pipeline that evaluates predictions on unmodified data. These pipelines will typically use different Dataset objects that have the same structure (i.e. the same types and compatible shapes for each component).
"""


