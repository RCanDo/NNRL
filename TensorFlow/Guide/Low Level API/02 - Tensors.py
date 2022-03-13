#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: Tensors
part: 1 
subtitle:
version: 1.0
type: code chunks
keywords: [TensorFlow, low level API]
description: Examples.
sources:
    - title: Tensors
      link: https://www.tensorflow.org/guide/tensors
      usage: copy
file:
    usage: 
        interactive: True
        terminal: False
    name: "02 - Tensors.py"
    path: "D:/ROBOCZY/Python/TensorFlow/Guide/Low Level API/"
    date: 2018-08-24
    authors:   
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: 
              - arkadiusz.kasprzyk@tieto.com
              - akasp666@google.com
              - arek@staart.pl      
"""  
#%%
#%%
"""
Tensors

TensorFlow, as the name indicates, is a framework to define and run computations involving tensors. 
A tensor is a generalization of vectors and matrices to potentially higher dimensions. 
Internally, TensorFlow represents tensors as n-dimensional arrays of base datatypes.

When writing a TensorFlow program, the main object you manipulate and pass around is the tf.Tensor. 
A tf.Tensor object represents a partially defined computation that will eventually produce a value. 
TensorFlow programs work by first building a graph of tf.Tensor objects, 
detailing how each tensor is computed based on the other available tensors 
and then by running parts of this graph to achieve the desired results.

A tf.Tensor has the following properties:
    a data type (float32, int32, or string, for example)
    a shape
Each element in the Tensor has the same data type, and the data type is always known. 
The shape (that is, the number of dimensions it has and the size of each dimension) 
might be only partially known. 
Most operations produce tensors of fully-known shapes if the shapes of their inputs 
are also fully known, but in some cases it's only possible to find the shape of a tensor 
at graph execution time.

Some types of tensors are special, and these will be covered in other units of the TensorFlow guide.
The main ones are:
    tf.Variable
    tf.constant
    tf.placeholder
    tf.SparseTensor
With the exception of tf.Variable, the value of a tensor is _immutable_, 
which means that in the context of a single execution tensors only have a single value. 
However, evaluating the same tensor twice can return different values; 
for example that tensor can be the result of reading data from disk, or generating a random number.
"""

#%%
cd "d:\ROBOCZY\Python\TensorFlow\Guide\Low Level API\"
pwd
ls

#%%

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

#%%
"""
Anaconda Prompt
>d:
>cd "ROBOCZY\Python\TensorFlow\Guide\Low Level API\"
remove all the files  `event*`  from current dir
>del event*
"""

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())

"""
> tensorboard --logdir ./ --host 127.0.0.1
in browser open:  http://127.0.0.1:6006/
"""

#%%
#init = tf.global_variables_initializer()

def run_and_show(lst):

    init = tf.global_variables_initializer()

    with tf.Session() as ss:
        ss.run(init)
        for tensor in lst:
            print(ss.run(tensor))
            print(ss.run(tf.rank(tensor)))   # ~= np.ndim()  # np.rank() depricated
            print(ss.run(tf.shape(tensor)))

    writer.add_graph(tf.get_default_graph())

#%% rank 0  <=>  shape = ()
mammal = tf.Variable("Elephant", tf.string)
ignition = tf.Variable(451, tf.int16)
pi = tf.Variable(3.14159265359, tf.float64)
cmplx = tf.Variable(12.3 - 4.85j, tf.complex64)

run_and_show([mammal, ignition, pi, cmplx])


#%% rank 1  <=>  shape = (1,)
v1 = tf.Variable([1], tf.int8)
v2 = tf.Variable(list(range(6)), tf.int8)
vs1 = tf.Variable(["sth"], tf.int8)
vs2 = tf.Variable(list("qqryq"), tf.int8)

run_and_show([v1, v2, vs1, vs2])


#%% rank 2  <=>  shape = (x, x)

mat1 = tf.Variable([[7],[11]], tf.int16)
mat2 = tf.Variable([[False, True],[True, False]], tf.bool)
matsq1 = tf.Variable([[4], [9], [16], [25]], tf.int32)
matsq2 = tf.Variable([ [4, 9], [16, 25] ], tf.int32)

run_and_show([mat1, mat2, matsq1, matsq2])


#%% 
# rank() is number of dimensions
rank_matsq1 = tf.rank(matsq1)   # it will show you rank of  matsq1
                                # only after graph runs 
rank_matsq1   # we see here rank of `rank_matsq1` not of `matsq1`  !!!

# while .shape is 'size' of each dimension
shape_matsq1 = tf.shape(matsq1)
shape_matsq1  # we see here shape of `shape_matsq1` not of `matsq1`  !!!
    
run_and_show([rank_matsq1, shape_matsq1])

#%%

my_image = tf.zeros([10, 299, 299, 3])  # batch x height x width x color

run_and_show([my_image])


#%%
# Referring to tf.Tensor slices

scalar1 = v1[0]
scalar2 = v2[1]
scalar3 = mat1[1, 0]

bool1 = mat2[1, 1]
vec1 = mat2[1,]
vec2 = matsq2[:, 1]
vec3 = matsq2[1]

vec4 = my_image[:, 2, 3:4, :]

run_and_show([scalar1, scalar2, scalar3, bool1, vec1, vec2, vec3, vec4])

"""
Note that the index passed inside the [] can itself be a scalar `tf.Tensor`,
if you want to dynamically choose an element from the vector.
"""

#%%

my_matrix = np.array([[1, 2, 3], [4, 5, 6]])
zeros = tf.zeros(my_matrix.shape[1])

run_and_show([zeros])

#%%

tfones = tf.ones([3, 4, 5])
matrix = tf.reshape(tfones, [6, 10])  # Reshape existing content into
                                      # a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  # Reshape existing content into a 3x20
                                       # matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
                                             # 4x3x5 tensor

run_and_show([tfones, matrix, matrixB, matrixAlt])

# Note that the number of elements of the reshaped Tensors has to match the
# original number of elements. Therefore, the following example generates an
# error because no possible value for the last dimension will match the number
# of elements.
yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # ERROR!


#%%  Data types

"""
It is not possible to have a tf.Tensor with more than one data type. 
It is possible, however, to _serialize_ arbitrary data structures as _strings_
and store those in `tf.Tensors`.
"""

# Cast a constant integer tensor into floating point.
float_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
run_and_show([float_tensor])

"""
To inspect a tf.Tensor's data type use the Tensor.dtype property.
"""

tfones.dtype
matrix.dtype
mammal.dtype

"""
When creating a tf.Tensor from a python object you may optionally specify the datatype. 
If you don't, TensorFlow chooses a datatype that can represent your data. 
TensorFlow converts Python integers to tf.int32 and python floating point numbers to tf.float32. 
Otherwise TensorFlow uses the same rules `numpy` uses when converting to arrays.
"""


#%%
# Evaluating tensors

constant = tf.constant([1, 2, 3])
tensor = constant * constant
print(tensor.eval())
"""
The eval method only works when a default tf.Session is active
(see Graphs and Sessions for more information).

Tensor.eval returns a _numpy_ array with the same contents as the tensor.
"""

sess = tf.Session()

print(sess.run(tensor.eval()))  # ERROR

print(tensor.eval(session=sess))
print(tensor.eval(session=sess))
print(tensor.eval())            # ERROR

# or

with sess.as_default():     # OK
    print(tensor.eval())

with tf.Session():    # OK
    print(tensor.eval())

#%%

p = tf.placeholder(tf.float32)
t = p + 1.
t.eval()  # This will fail, since the placeholder did not get a value.
print(t.eval(session=sess, feed_dict={p:2.}))

#!!!  Note that it is possible to feed any tf.Tensor, not just placeholders.

#%%
# Printing Tensors

"""
For debugging purposes you might want to print the value of a tf.Tensor.
While tfdbg provides advanced debugging support,
TensorFlow also has an operation to directly print the value of a tf.Tensor.

Note that you rarely want to use the following pattern when printing a tf.Tensor:
"""

t = <<some tensorflow operation>>
print(t)  # This will print the symbolic tensor when the graph is being built.
          # This tensor does not have a value in this context.


"""
This code prints the tf.Tensor object (which represents deferred computation)
and not its value.
Instead, TensorFlow provides the tf.Print operation,
which returns its first tensor argument unchanged
while printing the set of tf.Tensors it is passed as the second argument.

To correctly use tf.Print its return value must be used. See the example below
"""

t = <<some tensorflow operation>>
tf.Print(t, [t])  # This does nothing
t = tf.Print(t, [t])  # Here we are using the value returned by tf.Print
result = t + 1  # Now when result is evaluated the value of `t` will be printed.

"""
When you evaluate result you will evaluate everything result depends upon.
Since result depends upon t, and evaluating t has the side effect of printing
its input (the old value of t), t gets printed.
"""

#%%
# 


