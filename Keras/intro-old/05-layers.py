# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 11:56:31 2019

https://keras.io/layers/about-keras-layers/

author: rcando
email: rcando@int.pl
"""

"""
About Keras layers
------------------

All Keras layers have a number of methods in common:

    layer.get_weights():        returns the weights of the layer as a list of Numpy arrays.
    layer.set_weights(weights): sets the weights of the layer from a list of Numpy arrays
                                (with the same shapes as the output of get_weights).
    layer.get_config():         returns a dictionary containing the configuration of the layer.
                                The layer can be reinstantiated from its config via:

"""
layer = Dense(32)
config = layer.get_config()
reconstructed_layer = Dense.from_config(config)

# Or:

from keras import layers

config = layer.get_config()
layer = layers.deserialize({'class_name': layer.__class__.__name__,
                            'config': config})

"""
If a layer has a single node (i.e. if it isn't a shared layer), you can get its input tensor,
output tensor, input shape and output shape via:

    layer.input
    layer.output
    layer.input_shape
    layer.output_shape

If the layer has multiple nodes
(see: the concept of layer node and shared layers
https://keras.io/getting-started/functional-api-guide/#multi-input-and-multi-output-models
), you can use the following methods:

    layer.get_input_at(node_index)
    layer.get_output_at(node_index)
    layer.get_input_shape_at(node_index)
    layer.get_output_shape_at(node_index)
"""
