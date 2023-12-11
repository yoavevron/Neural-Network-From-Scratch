import numpy as np
from numpy import random


''' Random Gaussian : mean=0, standard deviation=1. widely used and often works well.
    He: Gaussian distribution with mean=0 and variance=2/n_in. effective for relu activation function.
    Xavier: Gaussian mean=0 and variance=2/(n_in + n_out). effective for sigmoid activation function.
    '''


def random_weights_init(input_size, hidden_layers, output_size):
    """
    Sets initiated random float value for the weights according to layers size with normal gaussian distribution
    :param input_size: Input layer size (weight x height of the input image)
    :param hidden_layers: tuple that represent the hidden layers. for example (16,16)
    :param output_size: Size of the output layer
    :return: A list of 2d np arrays that represent the weights between the layers
    """
    weights = []
    layer = random.randn(input_size, hidden_layers[0])
    weights.append(layer)
    for i in range(len(hidden_layers[:-1])):
        layer = random.randn(hidden_layers[i], hidden_layers[i+1])
        weights.append(layer)
    layer = random.randn(hidden_layers[-1], output_size)
    weights.append(layer)
    return weights


def random_bias_init(hidden_layers, output_size):
    """
    Sets initiated random float values for the biases according to layers size with normal gaussian distribution
    :param hidden_layers: tuple that represent the hidden layers. for example (16,16)
    :param output_size: Size of the output layer
    :return: A list of 1d np arrays that represent the biases between the layers
    """
    biases = []
    for i in range(len(hidden_layers)):
        # layer = random.randn(hidden_layers[i])
        layer = np.zeros(hidden_layers[i])
        biases.append(layer)
    # layer = random.randn(output_size)
    layer = np.zeros(output_size)
    biases.append(layer)
    return biases


def load_neurons(img, hidden_layers, output_layer):
    """
    Create a list of np arrays that will represent the neurons of the model.
    first layer is image pixels than the others are zeros in according to layers size.
    :param img: img pixels list
    :param hidden_layers: a tuple that represent the size of the hidden layers
    :param output_layer: a scalar that represent the output layer size
    :return: a list of 1d np arrays of the neurons for the model as described
    """
    neurons = []
    layer = img/255.0
    neurons.append(layer)
    for i in range(len(hidden_layers)):
        layer = np.zeros((1, hidden_layers[i]))
        neurons.append(layer)
    layer = np.zeros((1, output_layer))
    neurons.append(layer)
    return neurons


''' Implement '''


def he_weights_init(input_size, hidden_layers, output_size):
    return 1


def he_bias_init(hidden_layers, output_size):
    return 1


def xavier_weights_init(input_size, hidden_layers, output_size):
    # np_array = [len(layers)-1]
    # for i in range(len(np_array)-1):
    #     np_array[i] = [layers[i], layers[i+1]]
    # 	np_array.values = smart_random_values
    # 	return np_array
    return 1


def xavier_bias_init(hidden_layers, output_size):
    return 1
