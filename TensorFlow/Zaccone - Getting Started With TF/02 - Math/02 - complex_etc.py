#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: Complex, Random, PDE, etc
part: 2
subtitle:
version: 1.0
type: code chunks
keywords: [TensorFlow, math, complex numbers, random numbers, gradients, differential equations]
description: examples 
sources:
    - title: Getting Started with TensorFlow - Giancarlo Zaccone, 2016 
      chapter: 2. Doing Math with TensorFlow
      pages: 48-66
      link: "D:/bib/Python/TensorFlow/Getting Started With TensorFlow (178).pdf"
      date: 2016
      authors: 
          - fullname: Giancarlo Zaccone
      usage: code chunks copy and other examples
file:
    usage: 
        interactive: True
        terminal: False
    name: "01 - complex_etc.py"
    path: "D:/ROBOCZY/Python/TensorFlow/Zaccone - Getting Started With TF/02 - Math/"
    date: 2019-02-24
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

%reset

#%%

import keras.backend as K
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%%  Complex numbers and fractals
#%%

x = 5. + 4j
x

x = complex(5, 4)
x

x.real
x.imag

j     # NameError: name 'j' is not defined
1j
1j ** 2

#%%
#%%  Mandelbrot set

Y, X = np.mgrid[-1.3:1.3:.001, -2:1:.001]
type(Y)
Y.shape
X.shape
np.unique(X)
np.unique(Y)

Z = X + 1j*Y
type(Z)
Z[:5, :5]
c = tf.constant(Z, dtype='complex64')
c
# or
c = tf.constant(Z.astype(np.complex64))
c

zs = tf.Variable(c)
zs

ns = tf.Variable(tf.zeros_like(c, tf.float32))
ns

#%% build and execute the Data Flow Graph

ss = tf.InteractiveSession()
tf.global_variables_initializer().run()

# basic iteration
zs_ = zs*zs + c

# stop condition
not_diverged = tf.abs(zs_) < 4

# grouping multiple operations
step = tf.group(zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, tf.float32)))

for i in range(200): step.run()

# this may last quite a minute

#%% visualise the result

plt.imshow(ns.eval())
plt.show

#%%
#%%  Julia set
ss.close()
ss = tf.InteractiveSession()

Y, X = np.mgrid[-2:2:.002, -2:2:.002]
Z = X + 1j*Y
Z = tf.constant(Z.astype("complex64"))
Z.eval()

zs = tf.Variable(Z)
zs.eval()            # FailedPreconditionError: Attempting to use uninitialized value Variable

ns = tf.Variable(tf.zeros_like(Z, "float32"))

ss = tf.InteractiveSession()
tf.global_variables_initializer().run()

c = complex(0., .75)
zs_ = zs*zs - c

not_diverged = tf.abs(zs_) < 4

step = tf.group(zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, "float32")))

for i in range(200): step.run()

#%%
plt.imshow(ns.eval())
plt.show()


#%% 
#%%  Gradient

x = tf.placeholder(tf.float32)

y = 2 * x ** 2

var_grad = tf.gradients(y, x)

ss = tf.Session()

ss.run(var_grad, feed_dict={x:1})

# but
var_grad.eval(feed_dict={x:1})   # AttributeError: 'list' object has no attribute 'eval'

    
#%%
    
y = tf.placeholder(tf.float32)

z = tf.sqrt(x**2 + y**2)

var_grad = tf.gradients(z, (x, y))

ss.run(var_grad, feed_dict={x:1, y:2})

ss.run(var_grad, feed_dict={x:1, y:0})


#%%
#%% Random Numbers
"""
random_uniform(shape, minval, maxval, dtype, seed, name)
"""

import TensorFlow as tf
import matplotlib.pyplot as plt

uniform = tf.random_uniform([100], minval=0, maxval=1, dtype=tf.float32)

with tf.Session() as ss:
    u = uniform.eval()
    print(u)
    plt.hist(u, normed=True)
    plt.show()
    
#%%
normal = tf.random_normal([100], mean=0, stddev=2)
with tf.Session() as ss:
    u = normal.eval()
    print(u)
    plt.hist(u, normed=True)
    plt.show()
    
#%% Generating random numbers with seeds

uniform_with_seed = tf.random_uniform([1], seed=1)
uniform_without_seed = tf.random_uniform([1])

with tf.Session() as s1:
    print("seed = 1: {}".format(s1.run(uniform_with_seed)))
    print("seed = 1: {}".format(s1.run(uniform_with_seed)))
    print("no seed: {}".format(s1.run(uniform_without_seed)))
    print("no seed: {}".format(s1.run(uniform_without_seed)))

with tf.Session() as s1:
    print("seed = 1: {}".format(s1.run(uniform_with_seed)))
    print("seed = 1: {}".format(s1.run(uniform_with_seed)))
    print("no seed: {}".format(s1.run(uniform_without_seed)))
    print("no seed: {}".format(s1.run(uniform_without_seed)))

#%% Montecarlo's method
    
x = tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float32)
y = tf.random_uniform([1], minval=-1, maxval=1, dtype=tf.float32)

pi = list()

ss = tf.Session()

with ss.as_default():    #!!!
    n = 0
    k = 0
    for i in range(1000):
        n += 1
        hit = x.eval()**2 + y.eval()**2 < 1
        k += hit
        pi.append(4*k/n)

plt.plot(pi)
plt.show()

#%%
#%% Solving partial differential equations (PDE)
# see the problem in  https://www.TensorFlow.org/versions/r0.8/tutorials/pdes/index.html
"""the surface of square pond with a few raindrops landing on it;
the effect will be to produce bi-dimensional waves on the pond itself;
"""
    
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#%%

N = 500

u_init = np.zeros([N, N], dtype=np.float32)

# random droplets
for n in range(40):
    a, b = np.random.randint(0, N, 2)
    u_init[a, b] = np.random.uniform()
    
plt.imshow(u_init)    # ???
#plt.show()
    
#%%

ut_init = np.zeros([N, N], dtype=np.float32)
    
eps = tf.placeholder(tf.float32, shape=())
damping = tf.placeholder(tf.float32, shape=())

U = tf.Variable(u_init)
Ut = tf.Variable(ut_init)


def make_kernel(a):
    a = np.asarray(a)
    a = a.reshape(list(a.shape) + [1, 1])
    return tf.constant(a, dtype=1)

def simple_conv(x, k):
    x = tf.expand_dims(tf.expand_dims(x, 0), -1)   #???  reverse to  squeeze()
    y = tf.nn.depthwise_conv2d(x, k, [1, 1, 1, 1], padding='SAME')
    return y[0, :, :, 0]

def laplace(x):
    laplace_k = make_kernel([[.5, 1., .5], [1, -6., 1.], [.5, 1., .5]])
    return simple_conv(x, laplace_k)


U_ = U + eps * Ut
Ut_ = Ut + eps * (laplace(U) - damping * Ut)

step = tf.group(U.assign(U_), Ut.assign(Ut_))

#%% Graph execution
"""
In our session we will see the evolution in time of the pond by 1000 steps, where each time
step is equal to 0.03s, while the damping coefficient is set equal to 0.04.
"""

ss = tf.Session()


with ss.as_default():
    tf.initialize_all_variables().run()
    for i in range(1000):
        step.run({eps: .03, damping: .04})
        if i % 10 == 0:
            plt.clf()  # clearing currently active figure
            plt.imshow(U.eval())
            #plt.show()
        
#!!! something wrong with graphics:
# there is no animation
# figure refreshes onlny at the end
#            
#%%        





