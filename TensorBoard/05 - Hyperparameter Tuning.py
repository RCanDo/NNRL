#! python3
"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: Hyperparameter Tuning
part: 1 
subtitle:
version: 1.0
type: code chunks
keywords: [TensorBoard, Keras, MNIST, hyperparemeter tuning]
description: Examples.
sources:
    - title: Hyperparameter Tuning with the HParams Dashboard
      link: https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
      usage: copy
file:
    usage: 
        interactive: True
        terminal: False
    name: "05 - Hyperparameter Tuning.py"
    path: "D:/ROBOCZY/Python/TensorBoard/"
    date: 2019-10-12
    authors:   
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: 
              - arkadiusz.kasprzyk@tieto.com
              - akasp666@google.com
              - arek@staart.pl      
"""  
#%%
cd "D:/ROBOCZY/Python/TensorBoard/"
pwd

#%% Hyperparameter Tuning with the HParams Dashboard
"""
When building machine learning models, you need to choose various hyperparameters, 
such as the _dropout rate_ in a layer or the _learning rate_. 
These decisions impact model metrics, such as accuracy. 
Therefore, an important step in the machine learning workflow is 
to identify the best hyperparameters for your problem, which often involves experimentation. 
This process is known as "Hyperparameter Optimization" or "Hyperparameter Tuning".

The HParams dashboard in TensorBoard provides several tools to help with this process 
of identifying the best experiment or most promising sets of hyperparameters.

This tutorial will focus on the following steps:

    Experiment setup and HParams summary
    Adapt TensorFlow runs to log hyperparameters and metrics
    Start runs and log them all under one parent directory
    Visualize the results in TensorBoard's HParams dashboard

Note: The HParams summary APIs and dashboard UI are in a preview stage and will change over time.
"""
#%%
"""Start by installing TF 2.0 and loading the TensorBoard notebook extension:

!pip install -q tf-nightly-2.0-preview
# Load the TensorBoard notebook extension
%load_ext tensorboard

# Clear any logs from previous runs
(lin)  !rm -rf ./logs/
(win)  rmdir /q /s logs
"""
#%% Import TensorFlow and the TensorBoard HParams plugin:

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp

#%% Download the FashionMNIST dataset and scale it:

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#%% 1. Experiment setup and the HParams experiment summary
"""
Experiment with three hyperparameters in the model:

    Number of units in the first dense layer
    Dropout rate in the dropout layer
    Optimizer

List the values to try, and log an experiment configuration to TensorBoard. 
This step is optional: you can provide domain information 
to enable more precise filtering of hyperparameters in the UI, 
and you can specify which metrics should be displayed.
"""
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = hp.Metric('accuracy', display_name='Accuracy')

with tf.summary.create_file_writer('logs\\hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
    metrics=[METRIC_ACCURACY],       #!!! a way of evaluating (comparing) models
  )
  
"""
If you choose to skip this step, you can use a string literal 
wherever you would otherwise use an HParam value: e.g., 
hparams['dropout'] instead of hparams[HP_DROPOUT].
"""

#%% 2. Adapt TensorFlow runs to log hyperparameters and metrics
"""
The model will be quite simple: two dense layers with a dropout layer between them. 
The training code will look familiar, although the hyperparameters are no longer hardcoded. 
Instead, the hyperparameters are provided in an _hparams dictionary_ 
and used throughout the training function:
"""

def train_test_model(hparams):
  model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
  ])
  model.compile(
      optimizer=hparams[HP_OPTIMIZER],
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'],
  )

  model.fit(x_train, y_train, epochs=3)  # # Run with 1 epoch to speed things up for demo purposes
  _, accuracy = model.evaluate(x_test, y_test)
  return accuracy

#%% For each run, log an hparams summary with the hyperparameters and final accuracy:

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_test_model(hparams)
    tf.summary.scalar('accuracy', accuracy, step=1)

#%%
"""
When training Keras models, you can use callbacks instead of writing these directly:
    
model.fit(
    ...,
    callbacks=[
        tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        hp.KerasCallback(logdir, hparams),  # log hparams
    ],
)
"""

#%% 3. Start runs and log them all under one parent directory
"""
You can now try multiple experiments, training each one with a different set of hyperparameters.

For simplicity, use a grid search: 
try all combinations of the discrete parameters 
and just the lower and upper bounds of the real-valued parameter. 

For more complex scenarios, it might be more effective to choose each hyperparameter value randomly 
(this is called a random search). 
There are more advanced methods that can be used.

Run a few experiments, which will take a few minutes:
"""
session_num = 0

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
    for optimizer in HP_OPTIMIZER.domain.values:
      hparams = {
          HP_NUM_UNITS: num_units,
          HP_DROPOUT: dropout_rate,
          HP_OPTIMIZER: optimizer,
      }
      run_name = "run-%d" % session_num
      print('--- Starting trial: %s' % run_name)
      print({h.name: hparams[h] for h in hparams})
      run('logs\\hparam_tuning\\' + run_name, hparams)
      session_num += 1

#%% 4. Visualize the results in TensorBoard's HParams plugin
"""
The HParams dashboard can now be opened. 
Start TensorBoard and click on "HParams" at the top.

%tensorboard --logdir logs/hparam_tuning --host 127.0.0.1

The left pane of the dashboard provides filtering capabilities that are active 
across all the views in the HParams dashboard:

    Filter which hyperparameters/metrics are shown in the dashboard
    Filter which hyperparameter/metrics values are shown in the dashboard
    Filter on run status (running, success, ...)
    Sort by hyperparameter/metric in the table view
    Number of session groups to show (useful for performance when there are many experiments)

The HParams dashboard has three different views, with various useful information:

- The Table View lists the runs, their hyperparameters, and their metrics.
- The Parallel Coordinates View shows each run as a line going through an axis
  for each hyperparemeter and metric. 
  Click and drag the mouse on any axis to mark a region which will highlight 
  only the runs that pass through it. 
  This can be useful for identifying which groups of hyperparameters are most important. 
  The axes themselves can be re-ordered by dragging them.
- The Scatter Plot View shows plots comparing each hyperparameter/metric with each metric. 
  This can help identify correlations. 
  Click and drag to select a region in a specific plot and highlight those sessions 
  across the other plots.

???
A table row, a parallel coordinates line, and a scatter plot market 
can be clicked to see a plot of the metrics as a function of training steps for that session 
(although in this tutorial only one step is used for each run).
???

To further explore the capabilities of the HParams dashboard, 
download a set of pregenerated logs with more experiments:

%%bash
wget -q 'https://storage.googleapis.com/download.tensorflow.org/tensorboard/hparams_demo_logs.zip'
unzip -q hparams_demo_logs.zip -d logs/hparam_demo

View these logs in TensorBoard:

%tensorboard --logdir logs/hparam_demo

You can try out the different views in the HParams dashboard.

For example, by going to the parallel coordinates view 
and clicking and dragging on the accuracy axis, 
you can select the runs with the highest accuracy. 
As these runs pass through 'adam' in the optimizer axis, 
you can conclude that 'adam' performed better than 'sgd' on these experiments.
"""
