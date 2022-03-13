#! python3
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: tf.GradientTape
type: doc examples
sources:
    - link: https://www.tensorflow.org/api_docs/python/tf/GradientTape
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
import numpy as np
import tensorflow as tf

#%%
"""
Operations are recorded if they are executed within this _context manager_
and at least one of their inputs is being "watched".

Trainable variables (created by tf.Variable or tf.compat.v1.get_variable,
where `trainable=True` is default in both cases)
are automatically watched.
Tensors can be manually watched by invoking the .watch() method on this context manager.
"""

#%% For example, consider the function y = x * x. The gradient at x = 3.0 can be computed as:
x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  y = x * x
dy_dx = g.gradient(y, x) # Will compute to 6.0
print(dy_dx)
dy_dx.numpy()

#%% GradientTapes can be nested to compute higher-order derivatives. For example,

x = tf.constant(3.0)
with tf.GradientTape() as g:
  g.watch(x)
  with tf.GradientTape() as gg:
    gg.watch(x)
    y = x * x
  dy_dx = gg.gradient(y, x)     # Will compute to 6.0
  print(dy_dx)
d2y_dx2 = g.gradient(dy_dx, x)  # Will compute to 2.0
print(d2y_dx2)

#%% By default, the resources held by a GradientTape are released
#!!! as soon as GradientTape.gradient() method is called. !!!  (e.g. at (*) below)
# To compute multiple gradients over the same computation,
# create a persistent gradient tape. This allows multiple calls to the gradient() method
# as resources are released when the tape object is garbage collected. For example:

x = tf.constant(3.0)
with tf.GradientTape(persistent=True) as g:
  g.watch(x)
  y = x * x
  z = y * y
dz_dx = g.gradient(z, x)  # 108.0 (4*x^3 at x = 3)    (*)
print(dz_dx)
dy_dx = g.gradient(y, x)  # 6.0
print(dy_dx)
del g  # Drop the reference to the tape

#%% By default GradientTape will automatically watch any trainable variables
# that are accessed inside the context.
# If you want fine grained control over which variables are watched
# you can disable automatic tracking by passing `watch_accessed_variables=False`
# to the tape constructor:

p = tf.constant(3.)
q = tf.constant(4.)
with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
  g.watch(p)
  y = p ** 2  # Gradients will be available for `p`.
  z = q ** 3  # No gradients will be available since `q` is not being watched !
dy_dp = g.gradient(y, p)   # 6.
print(dy_dp)
dz_dq = g.gradient(z, q)   # None
print(dz_dq)

#%% Note that when using models you should ensure that your variables exist
# when using `watch_accessed_variables=False`.
# Otherwise it's quite easy to make your first iteration not have any gradients:

a = tf.keras.layers.Dense(32)
b = tf.keras.layers.Dense(32)

with tf.GradientTape(watch_accessed_variables=False) as tape:
  tape.watch(a.variables)  # Since `a.build` has not been called at this point
                           # `a.variables` will return an empty list and the
                           # tape will not be watching anything.
  result = b(a(inputs))
  tape.gradient(result, a.variables)  # The result of this computation will be
                                      # a list of `None`s since a's variables
                                      # are not being watched.

#%% Note that only tensors with real or complex dtypes are differentiable.

...
