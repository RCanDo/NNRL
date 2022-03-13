#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: Text classification with TensorFlow Hub
part: 1 
subtitle:
version: 1.0
type: code chunks
keywords: [TensorFlow, Keras, ML, image classification]
description: Examples.
sources:
    - title: Text classification with TensorFlow Hub: Movie reviews
      link: https://www.tensorflow.org/tutorials/keras/text_classification_with_hub
      usage: copy
file:
    usage: 
        interactive: True
        terminal: False
    name: "02 - Text Classification TF Hub.py"
    path: "D:/ROBOCZY/Python/TensorFlow2/Tutorials/ML With Keras/"
    date: 2019-10-04
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
This notebook classifies movie reviews as positive or negative using the text of the review. 
This is an example of binary — or two-class—classification, 
an important and widely applicable kind of machine learning problem.

The tutorial demonstrates the basic application of transfer learning with TensorFlow Hub and Keras.

We'll use the IMDB dataset that contains the text of 50,000 movie reviews 
from the Internet Movie Database. 
These are split into 25,000 reviews for training and 25,000 reviews for testing. 
The training and testing sets are balanced, meaning they contain an equal number 
of positive and negative reviews.

This notebook uses tf.keras, a high-level API to build and train models in TensorFlow, 
and TensorFlow Hub, a library and platform for transfer learning. 
For a more advanced text classification tutorial using tf.keras, 
see the [MLCC Text Classification Guide](https://developers.google.com/machine-learning/guides/text-classification/).
"""
#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

#%% Download the IMDB dataset
"""
The IMDB dataset is available on imdb reviews or on TensorFlow datasets. 
The following code downloads the IMDB dataset to your machine (or the colab runtime):
"""
# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(
    name="imdb_reviews", 
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)


#%% Explore the data
"""
Let's take a moment to understand the format of the data. 
Each example is a sentence representing the movie review and a corresponding label. 
The sentence is not preprocessed in any way. 
The label is an integer value of either 0 or 1, where 0 is a negative review, 
and 1 is a positive review.
"""

# Let's print first 10 examples.
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
train_examples_batch

# Let's also print the first 10 labels.
train_labels_batch

#%% Build the model
"""
The neural network is created by stacking layers—this requires three main architectural decisions:

    How to represent the text?
    How many layers to use in the model?
    How many hidden units to use for each layer?

In this example, the input data consists of sentences. The labels to predict are either 0 or 1.

One way to represent the text is to convert sentences into embeddings vectors. 
We can use a pre-trained _text embedding_ as the first layer, 
which will have three advantages: 
    * we don't have to worry about text preprocessing, 
    * we can benefit from _transfer learning_, 
    * the embedding has a fixed size, so it's simpler to process.

For this example we will use a pre-trained text embedding model from TensorFlow Hub 
called `google/tf2-preview/gnews-swivel-20dim/1`.

There are three other pre-trained models to test for the sake of this tutorial:

1. `google/tf2-preview/gnews-swivel-20dim-with-oov/1` 
    - same as `google/tf2-preview/gnews-swivel-20dim/1`, 
      but with 2.5% vocabulary converted to OOV buckets. 
      This can help if vocabulary of the task and vocabulary of the model don't fully overlap.
2. `google/tf2-preview/nnlm-en-dim50/1` 
    - A much larger model with ~1M vocabulary size and 50 dimensions.
3. `google/tf2-preview/nnlm-en-dim128/1`
    - Even larger model with ~1M vocabulary size and 128 dimensions.

Let's first create a Keras layer that uses a TensorFlow Hub model to embed the sentences, 
and try it out on a couple of input examples. 
Note that no matter the length of the input text, the output shape of the embeddings is: 
    (num_examples, embedding_dimension).
"""
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

#%% Let's now build the full model:

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

#%% 
"""
The layers are stacked sequentially to build the classifier:

1. The first layer is a TensorFlow Hub layer. 
   This layer uses a pre-trained Saved Model to map a sentence into its embedding vector.
   The pre-trained text embedding model that we are using `google/tf2-preview/gnews-swivel-20dim/1` 
   splits the sentence into tokens, embeds each token and then combines the embedding. 
   The resulting dimensions are: (num_examples, embedding_dimension).
2. This fixed-length output vector is piped through a fully-connected (Dense) layer 
   with 16 hidden units.
3. The last layer is densely connected with a single output node. 
   Using the sigmoid activation function, this value is a float between 0 and 1, 
   representing a probability, or confidence level.

Let's compile the model.
"""

#%% Loss function and optimizer
"""
A model needs a loss function and an optimizer for training. 
Since this is a binary classification problem and the model outputs a probability 
(a single-unit layer with a sigmoid activation), we'll use the  binary_crossentropy  loss function.

This isn't the only choice for a loss function, you could, for instance, choose mean_squared_error. 
But, generally, binary_crossentropy is better for dealing with probabilities
— it measures the "distance" between probability distributions, 
or in our case, between the ground-truth distribution and the predictions.

Later, when we are exploring regression problems (say, to predict the price of a house), 
we will see how to use another loss function called mean squared error.

Now, configure the model to use an optimizer and a loss function:
"""

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#%% Train the model
"""
Train the model for 20 epochs in mini-batches of 512 samples. 
This is 20 iterations over all samples in the x_train and y_train tensors. 
While training, monitor the model's loss and accuracy on the 10,000 samples from the validation set:
"""
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

#%% Evaluate the model
"""
And let's see how the model performs. 
Two values will be returned. 
Loss (a number which represents our error, lower values are better), and accuracy.
"""

results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
  print("%s: %.3f" % (name, value))
  
"""
49/49 - 3s - loss: 0.3178 - accuracy: 0.8640
loss: 0.318
accuracy: 0.864

This fairly naive approach achieves an accuracy of about 87%. 
With more advanced approaches, the model should get closer to 95%.
"""

#%% Further reading
"""
For a more general way to work with string inputs and for a more detailed analysis 
of the progress of accuracy and loss during training, take a look here.
"""


