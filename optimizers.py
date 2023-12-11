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


def adam():
    return 1
