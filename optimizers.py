import activation_functions
import numpy as np
import activation_functions


def gradient_descent(a, z, w, b, x, label, m, learn_rate):
    # How is it that all biases same, is it okay?
    dw_final = []
    db_final = []
    y = np.zeros(a[-1].shape)
    for i in range(len(y)):
        y[i][label[i]] = 1
    # verify functions and matrix doting
    old_dz = a[-1] - y
    dw = 1 / m * old_dz.T.dot(a[-2])
    db = np.full(a[-1].shape[1], 1 / m * np.sum(old_dz))
    dw_final.insert(0, dw.T)
    db_final.insert(0, db.T)

    for i in range(len(w)-2):
        index = -2-i
        new_dz = w[index+1].dot(old_dz.T) * activation_functions.relu_derivative(z[index].T)
        dw = 1 / m * new_dz.dot(a[index])
        db = np.full(a[i+1].shape[1], 1 / m * np.sum(new_dz))
        dw_final.insert(0, dw.T)
        db_final.insert(0, db.T)
        old_dz = new_dz
    new_dz = w[1].dot(old_dz) * activation_functions.relu_derivative(z[1].T)
    dw = 1 / m * new_dz.dot(x)
    db = np.full(a[1].shape[1], 1 / m * np.sum(new_dz))
    dw_final.insert(0, dw.T)
    db_final.insert(0, db.T)

    for i in range(len(db_final)):
        w[i] = w[i] - learn_rate * dw_final[i]
        b[i] = b[i] - learn_rate * db_final[i]

    return w, b


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
