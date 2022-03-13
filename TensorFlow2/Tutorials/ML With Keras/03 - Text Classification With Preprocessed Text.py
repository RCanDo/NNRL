#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: Text classification with preprocessed text
part: 1 
subtitle:
version: 1.0
type: code chunks
keywords: [TensorFlow, Keras, ML, image classification]
description: Examples.
sources:
    - title: Text classification with preprocessed text: Movie reviews
      link: https://www.tensorflow.org/tutorials/keras/text_classification?hl=en
      usage: copy
file:
    usage: 
        interactive: True
        terminal: False
    name: "03 - Text Classification With Preprocessed Text.py"
    path: "D:/ROBOCZY/Python/TensorFlow2/Tutorials/ML With Keras/"
    date: 2019-10-14
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
This is an example of binary—or two-class—classification, 
an important and widely applicable kind of machine learning problem.

We'll use the IMDB dataset that contains the text of 50,000 movie reviews 
from the Internet Movie Database. 
These are split into 25,000 reviews for training and 25,000 reviews for testing. 
The training and testing sets are balanced, 
meaning they contain an equal number of positive and negative reviews.

This notebook uses tf.keras, a high-level API to build and train models in TensorFlow. 
For a more advanced text classification tutorial using tf.keras, 
see the MLCC Text Classification Guide.
"""
#%% Setup

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow import keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np

print(tf.__version__)

2.0.0

Download the IMDB dataset

The IMDB movie reviews dataset comes packaged in tfds. It has already been preprocessed so that the reviews (sequences of words) have been converted to sequences of integers, where each integer represents a specific word in a dictionary.

The following code downloads the IMDB dataset to your machine (or uses a cached copy if you've already downloaded it):

To encode your own text see the Loading text tutorial

(train_data, test_data), info = tfds.load(
    # Use the version pre-encoded with an ~8k vocabulary.
    'imdb_reviews/subwords8k', 
    # Return the train/test datasets as a tuple.
    split = (tfds.Split.TRAIN, tfds.Split.TEST),
    # Return (example, label) pairs from the dataset (instead of a dictionary).
    as_supervised=True,
    # Also return the `info` structure. 
    with_info=True)

Downloading and preparing dataset imdb_reviews (80.23 MiB) to /home/kbuilder/tensorflow_datasets/imdb_reviews/subwords8k/0.1.0...
WARNING:tensorflow:From /home/kbuilder/.local/lib/python3.5/site-packages/tensorflow_datasets/core/file_format_adapter.py:209: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.
Instructions for updating:
Use eager execution and: 
`tf.data.TFRecordDataset(path)`

WARNING:tensorflow:From /home/kbuilder/.local/lib/python3.5/site-packages/tensorflow_datasets/core/file_format_adapter.py:209: tf_record_iterator (from tensorflow.python.lib.io.tf_record) is deprecated and will be removed in a future version.
Instructions for updating:
Use eager execution and: 
`tf.data.TFRecordDataset(path)`

Dataset imdb_reviews downloaded and prepared to /home/kbuilder/tensorflow_datasets/imdb_reviews/subwords8k/0.1.0. Subsequent calls will reuse this data.

Try the encoder

The dataset info includes the text encoder (a tfds.features.text.SubwordTextEncoder).

encoder = info.features['text'].encoder

print ('Vocabulary size: {}'.format(encoder.vocab_size))

Vocabulary size: 8185

This text encoder will reversibly encode any string:

sample_string = 'Hello TensorFlow.'

encoded_string = encoder.encode(sample_string)
print ('Encoded string is {}'.format(encoded_string))

original_string = encoder.decode(encoded_string)
print ('The original string: "{}"'.format(original_string))

assert original_string == sample_string

Encoded string is [4025, 222, 6307, 2327, 4043, 2120, 7975]
The original string: "Hello TensorFlow."

The encoder encodes the string by breaking it into subwords or characters if the word is not in its dictionary. So the more a string resembles the dataset, the shorter the encoded representation will be.

for ts in encoded_string:
  print ('{} ----> {}'.format(ts, encoder.decode([ts])))

4025 ----> Hell
222 ----> o 
6307 ----> Ten
2327 ----> sor
4043 ----> Fl
2120 ----> ow
7975 ----> .

Explore the data

Let's take a moment to understand the format of the data. The dataset comes preprocessed: each example is an array of integers representing the words of the movie review.

The text of reviews have been converted to integers, where each integer represents a specific word-piece in the dictionary.

Each label is an integer value of either 0 or 1, where 0 is a negative review, and 1 is a positive review.

Here's what the first review looks like:

for train_example, train_label in train_data.take(1):
  print('Encoded text:', train_example[:10].numpy())
  print('Label:', train_label.numpy())

Encoded text: [ 249    4  277  309  560    6 6639 4574    2   12]
Label: 1

The info structure contains the encoder/decoder. The encoder can be used to recover the original text:

encoder.decode(train_example)

"As a lifelong fan of Dickens, I have invariably been disappointed by adaptations of his novels.<br /><br />Although his works presented an extremely accurate re-telling of human life at every level in Victorian Britain, throughout them all was a pervasive thread of humour that could be both playful or sarcastic as the narrative dictated. In a way, he was a literary caricaturist and cartoonist. He could be serious and hilarious in the same sentence. He pricked pride, lampooned arrogance, celebrated modesty, and empathised with loneliness and poverty. It may be a cliché, but he was a people's writer.<br /><br />And it is the comedy that is so often missing from his interpretations. At the time of writing, Oliver Twist is being dramatised in serial form on BBC television. All of the misery and cruelty is their, but non of the humour, irony, and savage lampoonery. The result is just a dark, dismal experience: the story penned by a journalist rather than a novelist. It's not really Dickens at all.<br /><br />'Oliver!', on the other hand, is much closer to the mark. The mockery of officialdom is perfectly interpreted, from the blustering beadle to the drunken magistrate. The classic stand-off between the beadle and Mr Brownlow, in which the law is described as 'a ass, a idiot' couldn't have been better done. Harry Secombe is an ideal choice.<br /><br />But the blinding cruelty is also there, the callous indifference of the state, the cold, hunger, poverty and loneliness are all presented just as surely as The Master would have wished.<br /><br />And then there is crime. Ron Moody is a treasure as the sleazy Jewish fence, whilst Oliver Reid has Bill Sykes to perfection.<br /><br />Perhaps not surprisingly, Lionel Bart - himself a Jew from London's east-end - takes a liberty with Fagin by re-interpreting him as a much more benign fellow than was Dicken's original. In the novel, he was utterly ruthless, sending some of his own boys to the gallows in order to protect himself (though he was also caught and hanged). Whereas in the movie, he is presented as something of a wayward father-figure, a sort of charitable thief rather than a corrupter of children, the latter being a long-standing anti-semitic sentiment. Otherwise, very few liberties are taken with Dickens's original. All of the most memorable elements are included. Just enough menace and violence is retained to ensure narrative fidelity whilst at the same time allowing for children' sensibilities. Nancy is still beaten to death, Bullseye narrowly escapes drowning, and Bill Sykes gets a faithfully graphic come-uppance.<br /><br />Every song is excellent, though they do incline towards schmaltz. Mark Lester mimes his wonderfully. Both his and my favourite scene is the one in which the world comes alive to 'who will buy'. It's schmaltzy, but it's Dickens through and through.<br /><br />I could go on. I could commend the wonderful set-pieces, the contrast of the rich and poor. There is top-quality acting from more British regulars than you could shake a stick at.<br /><br />I ought to give it 10 points, but I'm feeling more like Scrooge today. Soak it up with your Christmas dinner. No original has been better realised."

Prepare the data for training

You will want to create batches of training data for your model. The reviews are all different lengths, so use padded_batch to zero pad the sequences while batching:

BUFFER_SIZE = 1000

train_batches = (
    train_data
    .shuffle(BUFFER_SIZE)
    .padded_batch(32, train_data.output_shapes))

test_batches = (
    test_data
    .padded_batch(32, train_data.output_shapes))

Each batch will have a shape of (batch_size, sequence_length) because the padding is dynaimic each batch will have a different langth:

for example_batch, label_batch in train_batches.take(2):
  print("Batch shape:", example_batch.shape)
  print("label shape:", label_batch.shape)
  

Batch shape: (32, 740)
label shape: (32,)
Batch shape: (32, 1355)
label shape: (32,)

Build the model

The neural network is created by stacking layers—this requires two main architectural decisions:

    How many layers to use in the model?
    How many hidden units to use for each layer?

In this example, the input data consists of an array of word-indices. The labels to predict are either 0 or 1. Let's build a "Continuous bag of words" style model for this problem:
Caution: This model doesn't use masking, so the zero-padding is used as part of the input, so the padding length may affect the output. To fix this, see the masking and padding guide.

model = keras.Sequential([
  keras.layers.Embedding(encoder.vocab_size, 16),
  keras.layers.GlobalAveragePooling1D(),
  keras.layers.Dense(1, activation='sigmoid')])

model.summary()

Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 16)          130960    
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 1)                 17        
=================================================================
Total params: 130,977
Trainable params: 130,977
Non-trainable params: 0
_________________________________________________________________

The layers are stacked sequentially to build the classifier:

    The first layer is an Embedding layer. This layer takes the integer-encoded vocabulary and looks up the embedding vector for each word-index. These vectors are learned as the model trains. The vectors add a dimension to the output array. The resulting dimensions are: (batch, sequence, embedding).
    Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each example by averaging over the sequence dimension. This allows the model to handle input of variable length, in the simplest way possible.
    This fixed-length output vector is piped through a fully-connected (Dense) layer with 16 hidden units.
    The last layer is densely connected with a single output node. Using the sigmoid activation function, this value is a float between 0 and 1, representing a probability, or confidence level.

Hidden units

The above model has two intermediate or "hidden" layers, between the input and output. The number of outputs (units, nodes, or neurons) is the dimension of the representational space for the layer. In other words, the amount of freedom the network is allowed when learning an internal representation.

If a model has more hidden units (a higher-dimensional representation space), and/or more layers, then the network can learn more complex representations. However, it makes the network more computationally expensive and may lead to learning unwanted patterns—patterns that improve performance on training data but not on the test data. This is called overfitting, and we'll explore it later.
Loss function and optimizer

A model needs a loss function and an optimizer for training. Since this is a binary classification problem and the model outputs a probability (a single-unit layer with a sigmoid activation), we'll use the binary_crossentropy loss function.

This isn't the only choice for a loss function, you could, for instance, choose mean_squared_error. But, generally, binary_crossentropy is better for dealing with probabilities—it measures the "distance" between probability distributions, or in our case, between the ground-truth distribution and the predictions.

Later, when we are exploring regression problems (say, to predict the price of a house), we will see how to use another loss function called mean squared error.

Now, configure the model to use an optimizer and a loss function:

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

Train the model

Train the model by passing the Dataset object to the model's fit function. Set the number of epochs.

history = model.fit(train_batches,
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)

Epoch 1/10
782/782 [==============================] - 7s 9ms/step - loss: 0.6826 - accuracy: 0.6205 - val_loss: 0.0000e+00 - val_accuracy: 0.0000e+00
Epoch 2/10
782/782 [==============================] - 5s 7ms/step - loss: 0.6219 - accuracy: 0.7525 - val_loss: 0.5929 - val_accuracy: 0.7906
Epoch 3/10
782/782 [==============================] - 5s 7ms/step - loss: 0.5420 - accuracy: 0.8060 - val_loss: 0.5277 - val_accuracy: 0.8156
Epoch 4/10
782/782 [==============================] - 5s 6ms/step - loss: 0.4754 - accuracy: 0.8394 - val_loss: 0.4748 - val_accuracy: 0.8417
Epoch 5/10
782/782 [==============================] - 5s 6ms/step - loss: 0.4219 - accuracy: 0.8618 - val_loss: 0.4340 - val_accuracy: 0.8615
Epoch 6/10
782/782 [==============================] - 5s 6ms/step - loss: 0.3802 - accuracy: 0.8766 - val_loss: 0.4010 - val_accuracy: 0.8635
Epoch 7/10
782/782 [==============================] - 5s 7ms/step - loss: 0.3503 - accuracy: 0.8850 - val_loss: 0.3779 - val_accuracy: 0.8698
Epoch 8/10
782/782 [==============================] - 5s 6ms/step - loss: 0.3250 - accuracy: 0.8929 - val_loss: 0.3630 - val_accuracy: 0.8625
Epoch 9/10
782/782 [==============================] - 5s 6ms/step - loss: 0.3044 - accuracy: 0.8994 - val_loss: 0.3447 - val_accuracy: 0.8729
Epoch 10/10
782/782 [==============================] - 5s 7ms/step - loss: 0.2876 - accuracy: 0.9046 - val_loss: 0.3281 - val_accuracy: 0.8875

Evaluate the model

And let's see how the model performs. Two values will be returned. Loss (a number which represents our error, lower values are better), and accuracy.

loss, accuracy = model.evaluate(test_batches)

print("Loss: ", loss)
print("Accuracy: ", accuracy)

782/782 [==============================] - 3s 4ms/step - loss: 0.3328 - accuracy: 0.8751
Loss:  0.33276922349124916
Accuracy:  0.87508

This fairly naive approach achieves an accuracy of about 87%. With more advanced approaches, the model should get closer to 95%.
Create a graph of accuracy and loss over time

model.fit() returns a History object that contains a dictionary with everything that happened during training:

history_dict = history.history
history_dict.keys()

dict_keys(['val_accuracy', 'val_loss', 'loss', 'accuracy'])

There are four entries: one for each monitored metric during training and validation. We can use these to plot the training and validation loss for comparison, as well as the training and validation accuracy:

import matplotlib.pyplot as plt

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

<Figure size 640x480 with 1 Axes>

plt.clf()   # clear figure

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()
