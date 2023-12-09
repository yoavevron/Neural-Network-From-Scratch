import activation_functions
import numpy as np


def stochastic_gradient_descent(img_pixels, neurons_before, neurons_after, weights, biases, real, learning_rate):
    real_output_vector = np.zeros(len(biases[-1]))
    real_output_vector[int(real)] = 1

    neurons_after[-1] = activation_functions.softmax(neurons_after[-1])

    previous_layer_error = (activation_functions.softmax_derivative
                            (neurons_after[-1]) *
                            neurons_after[-1] - real_output_vector)
    weights[-1] = weights[-1] - (learning_rate * np.outer(neurons_after[-2].T, previous_layer_error))
    biases[-1] = biases[-1] - (learning_rate * previous_layer_error)

    for i in range(1, len(neurons_after)-1):
        index = i - len(neurons_after)
        error_layer = (np.dot(previous_layer_error, weights[index+1].T) *
                       activation_functions.relu_derivative(neurons_after[index]))
        weights[index] = weights[index] - (learning_rate * np.outer(neurons_after[index].T, error_layer))
        biases[index] = biases[index] - (learning_rate * error_layer)
        previous_layer_error = error_layer

    first_layer_error = (np.dot(previous_layer_error, weights[1].T) *
                         activation_functions.relu_derivative(neurons_after[0]))
    weights[0] = weights[0] - (learning_rate * np.outer(img_pixels.T, first_layer_error))
    biases[0] = biases[0] - learning_rate * first_layer_error
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
