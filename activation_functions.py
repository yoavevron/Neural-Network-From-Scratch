import numpy as np


def relu(layer):  # Often used for continuous value classification
    return np.maximum(0, layer)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def softmax(layer):  # Often used for multiclass classification
    exp_x = np.exp(layer - np.max(layer))  # Subtracting np.max(x) for numerical stability
    return exp_x / np.sum(exp_x, axis=0)


def softmax_derivative(x):
    s = softmax(x)
    return s * (1 - s)


def sigmoid(num):  # Often used for binary classification
    return 0 or 1
