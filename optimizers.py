import activation_functions
import numpy as np


def stochastic_gradient_descent(neurons_before, neurons_after, weights, biases, real, learning_rate):
    real_output_vector = np.zeros(len(biases[-1]))
    real_output_vector[int(real)] = 1
    output_layer_error = (activation_functions.softmax_derivative
                          (activation_functions.softmax(neurons_before[-1])) *
                          neurons_after[-1] - real_output_vector)

    weights[-1] = weights[-1] - learning_rate * np.outer(neurons_after[-2].T, output_layer_error)
    biases[-1] = biases[-1] - learning_rate * output_layer_error

    return weights, biases


def adam():
    return 1


def backward_pass(X, y_true, hidden_inputs, hidden_outputs, weights, biases, learning_rate):
    real_output_vector = np.zeros(len(biases[-1]))
    real_output_vector[int(y_true)] = 1
    y_true = real_output_vector

    num_layers = len(hidden_inputs)

    # Compute the derivative of the loss with respect to the predicted output
    loss_derivative = np.dot(activation_functions.softmax_derivative(hidden_outputs[-1]), weights[-1]) * (hidden_outputs[-1] - y_true)
    print("g")
    print(learning_rate * np.dot(hidden_outputs[-2].T, loss_derivative))
    # Backpropagation for the output layer
    weights[-1] -= learning_rate * np.dot(hidden_outputs[-2].T, loss_derivative)
    biases[-1] -= learning_rate * np.sum(loss_derivative, axis=0)
    print()

    for i in range(num_layers - 1, 0, -1):
        # Backpropagation for hidden layers
        hidden_delta = (np.dot(loss_derivative, weights[i].T) *
                        activation_functions.relu_derivative(hidden_inputs[i - 1]))
        weights[i - 1] -= learning_rate * np.dot(hidden_outputs[i - 2].T, hidden_delta)
        biases[i - 1] -= learning_rate * np.sum(hidden_delta, axis=0)

        loss_derivative = hidden_delta

    # Backpropagation for the first hidden layer
    hidden_delta = np.dot(loss_derivative.T, weights[0].T) * activation_functions.relu_derivative(X)
    print(hidden_delta.shape)
    print(X.shape)
    print(hidden_delta.T.shape)
    weights[0] -= learning_rate * np.dot(X.reshape(1, len(X)), hidden_delta.T)
    biases[0] -= learning_rate * np.sum(hidden_delta, axis=0)

    return weights, biases
