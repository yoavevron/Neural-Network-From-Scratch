import numpy as np
import activation_functions


# Often used for classification
def categorical_cross_entropy(model_prediction_vector, real_output):
    """
    Gets the loss/cost between a predicted probabilities vector and a real output number
    :param model_prediction_vector: a 1d np array of model output probabilities for a result
    :param real_output: real output as a scalar
    :return: the loss between the input vector and the real output after converted to vector
    """
    return 1
    real_output_vector = np.zeros(model_prediction_vector.shape)
    for i in range(len(real_output_vector)):
        real_output_vector[i][real_output[i]] = 1
    real_output_vector[int(real_output)] = 1

    model_prediction_vector = activation_functions.softmax(model_prediction_vector)

    epsilon = 1e-15
    x = np.clip(model_prediction_vector, epsilon, 1 - epsilon)
    loss = -np.sum(real_output_vector * np.log(x))
    return loss


''' Implement '''


def mean_squared_error():  # Often used  for regression
    return 1
