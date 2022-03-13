"""
---  
# This is YAML, see: https://yaml.org/spec/1.2/spec.html#Preview
# !!! YAML message always begin with ---  

title: Get started with eager execution
part: 1 
subtitle:
version: 2.0
type: code chunks
keywords: [TensorFlow, eager execution]
description: examples 
sources:
    - link: https://www.tensorflow.org/get_started/eager
      usage: copy
file:
    usage: 
        interactive: True
        terminal: False
    name: "01 - Eager Execution.py"
    path: "D:/ROBOCZY/Python/TensorFlow/Get Started/"
    date: 2019-06-20
    authors:   
        - nick: kasprark
          fullname: Arkadiusz Kasprzyk
          email: 
              - arkadiusz.kasprzyk@tieto.com
              - akasp666@google.com
              - arek@staart.pl      
"""  


#%%

import tensorflow as tf;
import tensorflow.contrib.eager as tfe;


tf.enable_eager_execution();

iris_dataset_url = 'http://download.tensorflow.org/data/iris_training.csv';
iris_csv_file = tf.keras.utils.get_file('iris_dataset.csv', iris_dataset_url);

iris_dataset_tests_url = 'http://download.tensorflow.org/data/iris_test.csv';
iris_tests_csv_file = tf.keras.utils.get_file('iris_tests_dataset.csv', iris_dataset_tests_url);

#%%

def iris_data_parse_line(line):
    default_feature = [[0.0], [0.0], [0.0], [0.0], [0]]; #UPDATED SPOT!!!
    parsed_line = tf.decode_csv(line, default_feature);

    features = tf.reshape(parsed_line[:-1], shape=(4,), name="features");
    label = tf.reshape(parsed_line[-1], shape=(), name="label");

    return features, label;


def prediction_loss_diff(features, label, model):
    predicted_label = model(features);
    return tf.losses.sparse_softmax_cross_entropy(label, predicted_label);


def gradient_tune(features, targets, model):
    with tf.GradientTape() as tape:
        prediction_loss = prediction_loss_diff(features, targets, model);
    return tape.gradient(prediction_loss, model.variables);


def train_model(training_dataset, model, optimizer):
    train_loss_results = []
    train_accuracy_results = []
    rounds = 201;


    for round_num in range(rounds):
        epoch_loss_avg = tfe.metrics.Mean();
        epoch_accuracy = tfe.metrics.Accuracy();

        for features, label in training_dataset:
            gradients = gradient_tune(features, label, model);
            optimizer.apply_gradients(
                    zip(gradients, model.variables),
                    global_step=tf.train.get_or_create_global_step());


def main():
    print("TensorFlow version: {}".format(tf.VERSION));
    print("Eager execution: {}".format(tf.executing_eagerly()));

    iris_dataset = (tf.data.TextLineDataset(iris_csv_file)
                           .skip(1)
                           .map(iris_data_parse_line)
                           .shuffle(1000)
                           .batch(32));

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(3)
    ]);

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01);

    train_model(iris_dataset, model, optimizer);

#%%

if __name__ == "__main__":
    main();