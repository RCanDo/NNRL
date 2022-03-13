#! python3
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: tf.function
type: doc examples
sources:
    - link: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/function?hl=en
      usage: copy
file:
    usage:
        interactive: True
        terminal: False
    path: ".../Python/TensorFlow2/API/"
    date: 2019-10-03
    authors:
        - nick: RCanDo
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
"""
#%%
import tensorflow as tf

#%%
"""
function() constructs a callable that executes a TensorFlow graph (tf.Graph)
created by tracing the TensorFlow operations in func.
This allows the TensorFlow runtime to apply optimizations and exploit parallelism
in the computation defined by func.
"""

def f(x, y):
  return tf.reduce_mean(tf.multiply(x ** 2, 3) + y)

g = tf.function(f)

x = tf.constant([[2.0, 3.0]])
y = tf.constant([[3.0, -2.0]])

# `f` and `g` will return the same value, but `g` will be executed as a
# TensorFlow graph.
assert f(x, y).numpy() == g(x, y).numpy()

# Tensors and tf.Variables used by the Python function are captured in the
# graph.
@tf.function
def h():
  return f(x, y)

assert (h().numpy() == f(x, y).numpy()).all()

# Data-dependent control flow is also captured in the graph. Supported
# control flow statements include `if`, `for`, `while`, `break`, `continue`,
# `return`.
@tf.function
def g(x):
  if tf.reduce_sum(x) > 0:
    return x * x
  else:
    return -x // 2

g(x)

# print and TensorFlow side effects are supported, but exercise caution when
# using Python side effects like mutating objects, saving to files, etc.
l = []
v = tf.Variable([0., 0.])

@tf.function
def g(x):
  for i in x:
    print(i)                              # Works
    tf.compat.v1.assign(v, i)                       # Works
    tf.compat.v1.py_func(lambda i: l.append(i))(i)  # Works
    #l.append(i)                           # Caution! Doesn't work.

g(x)  # ???

#%%
"""
Note that unlike other TensorFlow operations,
!!! we don't convert python numerical inputs to tensors. !!!
Moreover, a new graph is generated for each distinct python numerical value,
for example calling g(2) and g(3) will generate two new graphs
(while only one is generated if you call g(tf.constant(2)) and g(tf.constant(3))).
Therefore, python numerical inputs should be restricted to arguments
that will have few distinct values,
such as hyperparameters like the number of layers in a neural network.
This allows TensorFlow to optimize each variant of the neural network.
"""

#%% Referencing tf.Variables
"""
The Python function func may reference _stateful_ objects (such as tf.Variable).
These are captured as implicit inputs to the callable returned by function. For example:
"""
...


