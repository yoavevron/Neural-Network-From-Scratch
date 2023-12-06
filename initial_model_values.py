import numpy as np


def load_neurons(img):
    neurons = np.zeros(img)
    return neurons


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


def ha_weights_init(input_size, hidden_layers, output_size):
    neurons = np.zeros(hidden_layers)
    weights = np.zeros(hidden_layers)
    bias = np.zeros(hidden_layers)
    return neurons, weights, bias


def xavier_bias_init(input_size, hidden_layers, output_size):
    neurons = np.zeros(hidden_layers)
    weights = np.zeros(hidden_layers)
    bias = np.zeros(hidden_layers)
    return neurons, weights, bias


def ha_bias_init(input_size, hidden_layers, output_size):
    neurons = np.zeros(hidden_layers)
    weights = np.zeros(hidden_layers)
    bias = np.zeros(hidden_layers)
    return neurons, weights, bias
