#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: Basic regression: Predict fuel efficiency
part: 1 
subtitle:
version: 1.0
type: code chunks
keywords: [TensorFlow, Keras, ML, regression]
description: Examples.
sources:
    - title: Basic regression: Predict fuel efficiency
      link: https://www.tensorflow.org/tutorials/keras/regression
      usage: copy
file:
    usage: 
        interactive: True
        terminal: False
    name: "04 - Basic Regression.py"
    path: "D:/ROBOCZY/Python/TensorFlow2/Tutorials/ML With Keras/"
    date: 2019-10-03
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
In a regression problem, we aim to predict the output of a continuous value, 
like a price or a probability. 
Contrast this with a classification problem, where we aim to select a class from a list of classes 
(for example, where a picture contains an apple or an orange, 
recognizing which fruit is in the picture).

This notebook uses the classic [Auto MPG Dataset](https://archive.ics.uci.edu/ml/datasets/auto+mpg) 
and builds a model to predict the fuel efficiency of late-1970s and early 1980s automobiles. 
To do this, we'll provide the model with a description of many automobiles from that time period. 
This description includes attributes like: cylinders, displacement, horsepower, and weight.

This example uses the tf.keras API, see this guide for details.
"""
#%%
from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
# https://stackoverflow.com/questions/11707586/how-do-i-expand-the-output-display-to-see-more-columns
pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)


# Use seaborn for pairplot
# !pip install -q seaborn
import seaborn as sns    # for pairplot

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

#%% The Auto MPG dataset
"""
The dataset is available from the 
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).
"""
#%% Get the data
# First download the dataset.

dataset_path = keras.utils.get_file("auto-mpg.data", 
                "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path

#%% Import it using pandas

column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']

raw_dataset = pd.read_csv(dataset_path)
raw_dataset  # not really what we want - we must look at the data at the site
             # to know variables names, etc.

raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
dataset.head()
dataset.tail()

type(dataset)  # pandas.core.frame.DataFrame
dataset.shape  # (398, 8)
dataset.describe().T
dataset.columns

#%% Clean the data
dataset.isna().sum()
"""
MPG             0
Cylinders       0
Displacement    0
Horsepower      6
Weight          0
Acceleration    0
Model Year      0
Origin          0
dtype: int64
"""

# To keep this initial tutorial simple drop those rows.
dataset = dataset.dropna()
dataset.shape

dataset['Cylinders'].value_counts()
dataset['Origin'].value_counts()

#%% The "Origin" column is really categorical, not numeric. So convert that to a one-hot:

origin = dataset.pop('Origin')
origin.value_counts()

dataset['USA'] = (origin == 1)*1.0
# dataset.loc[:, ('USA')] = (origin == 1)*1.0  # also complains...
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

#%% Split the data into train and test
# Now split the dataset into a training set and a test set.
# We will use the test set in the final evaluation of our model.

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#%% Inspect the data
# Have a quick look at the joint distribution of a few pairs of columns from the training set.

sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")

#%% Also look at the overall statistics:

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
train_stats

#%% Split features from labels
"""
Separate the target value, or "label", from the features. 
This label is the value that you will train the model to predict.
"""
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

#%% Normalize the data
"""
Look again at the train_stats block above and note how different the ranges of each feature are.

It is good practice to normalize features that use different scales and ranges. 
Although the model might converge without feature normalization, 
it makes training more difficult, and it makes the resulting model dependent 
on the choice of units used in the input.

Note: Although we intentionally generate these statistics from only the training dataset, 
these statistics will also be used to normalize the test dataset. 
We need to do that to project the test dataset into the same distribution 
that the model has been trained on.
"""
def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

"""
This normalized data is what we will use to train the model.
Caution: The statistics used to normalize the inputs here (mean and standard deviation) 
need to be applied to any other data that is fed to the model, 
along with the one-hot encoding that we did earlier. 
That includes the test set as well as live data when the model is used in production.
"""

#%% The model

#%% Build the model
"""
Let's build our model. 
Here, we'll use a Sequential model with two densely connected hidden layers, 
and an output layer that returns a single, continuous value. 
The model building steps are wrapped in a function, build_model, since we'll create a second model, 
later on.
"""

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model
  
model = build_model()

#%% Inspect the model
# Use the .summary method to print a simple description of the model

len(train_dataset.keys())   # 9

model.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 64)                640        (9 + 1)*64
_________________________________________________________________
dense_1 (Dense)              (None, 64)                4160       (64 + 1)*64
_________________________________________________________________
dense_2 (Dense)              (None, 1)                 65         
=================================================================
Total params: 4,865
Trainable params: 4,865
Non-trainable params: 0
_________________________________________________________________
"""

#%% Now try out the model. 
# Take a batch of 10 examples from the training data and call model.predict on it.

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result
# It seems to be working, and it produces a result of the expected shape and type.

# Of course, the real values are completely different, as model is not trained yet:
train_labels[:10]

#%% Train the model
# Train the model for 1000 epochs, 
# and record the training and validation accuracy in the _history object_.

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

#%%
    
EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

#%% Visualize the model's training progress using the stats stored in the history object.

type(history)  # tensorflow.python.keras.callbacks.History
dir(history)  
history.params
"""
{'batch_size': 32,
 'epochs': 1000,
 'steps': None,
 'samples': 251,   # 314 * .8   # normed_train_data.shape  # (314, 9)
 'verbose': 0,
 'do_validation': True,
 'metrics': ['loss', 'mae', 'mse', 'val_loss', 'val_mae', 'val_mse']}
"""
type(history.history)
history.history.keys()
len(history.history['loss'])  # 1000
len(history.epoch)            # 1000

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

#%%

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  
  plt.show()

plot_history(history)

#%% 
"""
This graph shows little improvement, or even degradation in the validation error 
after about 100 epochs. 
Let's update the model.fit call to automatically stop training 
when the validation score doesn't improve. 
We'll use an EarlyStopping callback that tests a training condition for every epoch. 
If a set amount of epochs elapses without showing improvement, then automatically stop the training.
"""

# we must build the model anew
model2 = build_model()

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history2 = model2.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history2)

"""
The graph shows that on the validation set, the _average error_ is usually around +/- 2 MPG. 
Is this good? We'll leave that decision up to you.
"""
#%%
"""
Let's see how well the model generalizes by using the test set, 
which we did not use when training the model. 
This tells us how well we can expect the model to predict when we use it in the real world.
"""
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

#%% Make predictions
# Finally, predict MPG values using data in the testing set:

test_predictions = model.predict(normed_test_data).flatten()

plt.figure()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])    # ???

#%% 
"""
It looks like our model predicts reasonably well. 
Let's take a look at the error distribution.
"""
plt.figure()
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

"""
It's not quite gaussian, but we might expect that because the number of samples is very small.
"""

#%% Conclusion
"""
This notebook introduced a few techniques to handle a regression problem.
1. Mean Squared Error (MSE) is a common loss function used for regression problems 
   (different loss functions are used for classification problems).
2. Similarly, evaluation metrics used for regression differ from classification. 
   A common regression metric is Mean Absolute Error (MAE).
3. When numeric input data features have values with different ranges, 
   each feature should be scaled independently to the same range.
4. If there is not much training data, one technique is to prefer a small network 
   with few hidden layers to avoid overfitting.
5. Early stopping is a useful technique to prevent overfitting.
"""
