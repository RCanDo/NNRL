#! python3
"""
---
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---

title: Usage of loss function
part: 1
subtitle:
version: 1.0
type: code chunks
keywords: [TensorBoard, Keras, loss function]
description: Examples.
sources:
    - title: TF2 API doc
      link: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/losses?hl=en
      usage: reference
file:
    usage:
        interactive: True
        terminal: False
    name: "01 - losses.py"
    path: ".../Python/TensorFlow2/My Notes/"
    date: 2019-10-10
    authors:
        - nick: rcando
          fullname: Arkadiusz Kasprzyk
          email:
              - rcando@int.pl
"""
#%%
cd "D:/ROBOCZY/Python/TensorFlow2/My Notes/"
pwd
ls

#%%
import tensorflow as tf


#%%
#%% my notes
#%%
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(10, activation="softmax")
        ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

history = model.fit(x=x_train, y=y_train, epochs=5, validation_data=(x_test, y_test))

#%% by the way
y_pred = model.predict(x_test)
y_pred.shape  # 10000, 10
y_pred[:5]

list(map(lambda y: y.argmax(), y_pred[:5]))
[y.argmax() for y in y_pred[:5]]
y_test[:5]

sum(list(map(lambda yy: yy[0].argmax() != yy[1] , zip(y_pred, y_test))))
sum(yy[0].argmax() != yy[1] for yy in zip(y_pred, y_test))
# 235  # may vary depending on how the model learned


#%% THE PROBLEM

#%% let

xx = x_test[:5]
type(xx)  # numpy.ndarray
xx.shape  # (5, 28, 28)

#%% !!! notice the difference:

model.predict(xx)  # numpy.ndarray
model.predict(xx).shape  # (5, 10)

model(xx)          # <tf.Tensor: id=30787, shape=(5, 10), dtype=float32, numpy=`model.predict(xx)`>
model(xx).shape    # TensorShape([5, 10])

# this is sth to rmember !!!

#%% ???!!!???

loss = tf.keras.losses.SparseCategoricalCrossentropy()
loss(y_test, y_pred)  #!!! AttributeError: 'list' object has no attribute 'op'

loss(y_test, model(x_test))  # <tf.Tensor: id=31254, shape=(), dtype=float32, numpy=0.073122285>
loss(y_test, model(x_test)).numpy()  # 0.073122285


#%% from TF2 API specs
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/losses/SparseCategoricalCrossentropy?hl=en
cce = tf.keras.losses.SparseCategoricalCrossentropy()
cce([0, 1, 2], [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
#!!! AttributeError: 'list' object has no attribute 'op'

cce([0, 1, 2], tf.constant([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]]))
   # <tf.Tensor: id=31327, shape=(), dtype=float32, numpy=0.32396814>
cce([0, 1, 2], tf.constant([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])).numpy()
   # 0.32396814

cce(tf.constant([0, 1, 2]), [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
   # AttributeError: 'list' object has no attribute 'op'
cce(tf.constant([0, 1, 2]), tf.constant([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]]))
   # OK!

"""HENCE:
The SECOND argument MUST be  tf.Tensor  (np.array is not enough)
"""

#%% the analogous function:
tf.keras.losses.sparse_categorical_crossentropy(
    [0, 1, 2],
    [[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]])
#!!! AttributeError: 'list' object has no attribute 'op'

#!!! THE SAME PROBLEM

tf.keras.losses.sparse_categorical_crossentropy(
    [0, 1, 2],
    tf.constant([[.9, .05, .05], [.5, .89, .6], [.05, .01, .94]]))
    # <tf.Tensor: id=31402, shape=(3,), dtype=float32,
    #  numpy=array([0.10536056, 0.8046684 , 0.0618754 ], dtype=float32)>
    # ???
    # quite a different output...

sum([0.10536056, 0.8046684 , 0.0618754 ])/3  # 0.32396812   OK

#%%

#%%

test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(1)
test_ds.take(1)  # <TakeDataset shapes: ((None, 28, 28), (None,)), types: (tf.float64, tf.uint8)>

for (x, y) in test_ds.take(1):
    print(type(x))  # ok
    print(y)  # ok
    print(model(x))
    yhat = model.predict(x)
    print("{} -- {}".format(y, yhat))
    print(loss(y,     model(x)))

#%% some figure?
