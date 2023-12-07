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
    real_output_vector = np.zeros(len(model_prediction_vector))
    real_output_vector[int(real_output)] = 1

    model_prediction_vector = activation_functions.softmax(model_prediction_vector)

    epsilon = 1e-15
    x = np.clip(model_prediction_vector, epsilon, 1 - epsilon)
    loss = -np.sum(real_output_vector * np.log(x)) / real_output_vector.shape[0]
    return loss


''' Implement '''


def mean_squared_error():  # Often used  for regression
    return 1
