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
keywords: [TensorFlow, math]
description: examples 
sources:
    - title: Getting Started with TensorFlow - Giancarlo Zaccone, 2016 
      chapter: 2. Doing Math with TensorFlow
      pages: 35-47
      link: "D:/bib/Python/TensorFlow/Getting Started With TensorFlow (178).pdf"
      date: 2016
      authors: 
          - fullname: Giancarlo Zaccone
      usage: code chunks copy and other examples
file:
    usage: 
        interactive: True
        terminal: False
    name: "01 - tensors.py" 
    path: "D:/ROBOCZY/Python/TensorFlow/Zaccone - Getting Started With TF/02 - Math/"
    date: 2019-02-05
    authors:   
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: 
              - arkadiusz.kasprzyk@tieto.com
              - akasp666@google.com
              - arek@staart.pl      
"""              


#%%

cd "D:\ROBOCZY\Python\TensorFlow\Zaccone - Getting Started With TF\02 - Math"
pwd
ls

#%%
import numpy as np
import tensorflow as tf


#%% p.37

mx1 = np.array([1.3, 1, 4.0, 23.99])
mx1.shape
mx1.ndim
mx1.dtype

#%% 

tr1 = tf.convert_to_tensor(mx1, dtype=tf.float32)
tr1
tf.Tensor(mx1, dtype=tf.float32)  # TypeError: __init__() missing 1 required positional argument: 'value_index'

with tf.Session() as ss:
    print(ss.run(tr1))
    print(ss.run(tr1[0]))
    print(ss.run(tf.shape(tr1)))
    print(ss.run(tf.rank(tr1)))
    # print(ss.run(tf.dtype(tr1))) AttributeError: module 'tensorflow' has no attribute 'dtype'
    # print(ss.run(tr1.dtype))   #! ERROR too..

with tf.Session() as ss:
    list(map(lambda x: print(ss.run(x)), [tr1, tr1[0], tf.shape(tr1), tf.rank(tr1)]))


with tf.Session() as ss:
    for k in map(lambda x: ss.run(x), [tr1, tr1[0], tf.shape(tr1), tf.rank(tr1)]):
        print(k)


#%%

mx2 = np.array([(2, 2, 2), (2, 2, 2), (2, 2, 2)], dtype='int32')
mx3 = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)], dtype='int32')

tx2 = tf.constant(mx2)
tx2

tx3 = tf.constant(mx3)
tx3

with tf.Session() as ss:
    for k in [tx2, tx3]: print(ss.run(k))

with tf.Session() as ss:
    [print(ss.run(k)) for k in [tx2, tx3]]

#%%

ss = tf.Session()

mx_prod = tf.matmul(mx2, mx3)
mx_prod
print(ss.run(mx_prod))

mx_sum = tf.add(mx2, mx3)
mx_sum

for k in [mx_prod, mx_sum]: print(ss.run(k))

#%%

mx_det = tf.matrix_determinant(tx2)  #!  TypeError: Value passed to parameter 'input' 
                                     #   has DataType int32 not in list of allowed values: 
                                     #   float32, float64, complex64, complex128

tx2 = tf.constant(mx2, dtype='float32')
tx2

mx_det = tf.matrix_determinant(tx2)  
ss.run(mx_det)

#%% tf operators

tf0 = tf.zeros((3, 3), dtype='float32')
tf1 = tf.ones((3, 3), dtype='float32')
tf2 = 2 * tf1
tf3 = 3 * tf1

tfi = tf.eye(3, 3)
tfr = tf.random_uniform((3, 3), dtype='float32')
ss.run(tfr)
ss.run(tfr)  # every time different result...
## but we need random constant

tfr = tf.constant( np.random.rand(3, 3), dtype='float32' )
ss.run(tfr)


# operators -- elementwise!

ss.run( tf.add(tf0, tf1) )

ss.run( tf.subtract(tf2, tf1) )
ss.run( tf.subtract(tf1, tfi) )

ss.run( tf.multiply(tfr, tf2) )
ss.run( tf.multiply(tfr, tfi) )  # elementwise multiplication 
ss.run( tf.matmul(tfr, tfi) )    # matrix multiplication

ss.run( tf.div(tf1, tf2) )      

ss.run( tf.mod(tf3, tf2) )

# ---

tfrn = tf.constant( np.random.randn(3, 3), dtype='float32' )
ss.run( tfrn )

ss.run( tf.abs(tfrn) )
ss.run( tf.negative(tfrn) )

# ---

ss.run( tf.sign(tfrn) )
ss.run( tf.matrix_inverse(tfrn) )

ss.run( tf.matrix_inverse(tfrn) * tfrn )   ## it's elementwise!!!
ss.run( tf.matmul( tf.matrix_inverse(tfrn), tfrn) )
ss.run( tf.matmul( tfrn, tf.matrix_inverse(tfrn)) )

# ---

ss.run( tf.square(tf2) )
ss.run( tf.round( 100*tfrn ) )

ss.run( tf.pow(tf2, 3) )
ss.run( tf.exp(tfrn) )
ss.run( tf.log(tfr) )

ss.run( tf... ( tfrn, tf.transpose(tfrn) ) )       # ?? how to concatenate matrices ???
ss.run( tf.maximum(tfrn, tf.transpose(tfrn)) )
ss.run( tf.minimum(tfrn, tf.transpose(tfrn)) )

ss.run( tf.cos(tfrn) )
ss.run( tf.sin(tfrn) )


#%%  3d
tsd = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
tsd
tsd.shape
tsd[0, 1, 0]


#%%
#%%  Prepare an input data

import matplotlib.image as mp_image
filename = "TheLadyWithAnErmine.jpg"
image = mp_image.imread(filename)

image.ndim
image.shape
image.size

#%%
import matplotlib.pyplot as plt

plt.imshow(image)
plt.show()

#%%  take only the slice from the image

image_tf = tf.placeholder("uint8", [None, None, 3])             #!!!

image_slice = tf.slice(input_=image_tf, begin=[240, 0, 0], size=[50, -1, -1], name=None)
# -1 in `size` means "take everything from the `begin` to the end (in given dimension)

ss = tf.Session()

def show_image_slice(image_slice):
    result = ss.run(image_slice, feed_dict={image_tf: image})
    print(result.shape)
    plt.imshow(result)
    plt.show()

show_image_slice(image_slice)

#%%  using tf.Tensor.getitem() i.e. pythonic slicing instead of tf.slice()
image_slice = image_tf[240:290, :, :]

show_image_slice(image_slice)

#%%  take only one basic color
image_slice = image_tf[ :, :, 0]

show_image_slice(image_slice)

#%%  permute colors
image_col = image_tf[ :, :, ::-1]

show_image_slice(image_col)

#%%  transformations

image_var = tf.Variable(image)
model = tf.global_variables_initializer()                       #!!!
 
image_trans = tf.transpose(image_var, perm=[1, 0, 2])

ss.run(model)
res = ss.run(image_trans)

plt.imshow(res)
plt.show()


#%%
#%%


