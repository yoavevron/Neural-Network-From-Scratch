import numpy as np
from numpy import random


def load_neurons(img):
    neurons = np.zeros(img)
    return neurons


''' Random Gaussian : mean=0, standard deviation=1. widely used and often works well.
    Xavier: Gaussian mean=0 and variance=2/(n_in + n_out). effective for sigmoid activation function.
    He: Gaussian distribution with mean=0 and variance=2/n_in. effective for relu activation function.'''


def random_weights_init(input_size, hidden_layers, output_size):
    """
    Sets initiated random float value for the weights according to layers size with normal gaussian distribution
    :param input_size: Input layer size (weight x height of the input image)
    :param hidden_layers: tuple that represent the hidden layers. for example (16,16)
    :param output_size: Size of the output layer
    :return: A list of 2d np arrays that represent the weights between the layers
    """
    weights = []
    layer = random.rand(input_size, hidden_layers[0])
    weights.append(layer)
    for i in range(len(hidden_layers[:-1])):
        layer = random.rand(hidden_layers[i], hidden_layers[i+1])
        weights.append(layer)
    layer = random.rand(hidden_layers[-1], output_size)
    weights.append(layer)
    return weights


def random_bias_init(input_size, hidden_layers, output_size):
    bias =[]
    return bias


def he_weights_init(input_size, hidden_layers, output_size):
    neurons = np.zeros(hidden_layers)
    weights = np.zeros(hidden_layers)
    bias = np.zeros(hidden_layers)
    return neurons, weights, bias


def he_bias_init(input_size, hidden_layers, output_size):
    neurons = np.zeros(hidden_layers)
    weights = np.zeros(hidden_layers)
    bias = np.zeros(hidden_layers)
    return neurons, weights, bias


def xavier_weights_init(input_size, hidden_layers, output_size):
    # np_array = [len(layers)-1]
    # for i in range(len(np_array)-1):
    #     np_array[i] = [layers[i], layers[i+1]]
    # 	np_array.values = smart_random_values
    # 	return np_array
    neurons = np.zeros(hidden_layers)
    weights = np.zeros(hidden_layers)
    bias = np.zeros(hidden_layers)
    return neurons, weights, bias


def xavier_bias_init(input_size, hidden_layers, output_size):
    neurons = np.zeros(hidden_layers)
    weights = np.zeros(hidden_layers)
    bias = np.zeros(hidden_layers)
    return neurons, weights, bias
