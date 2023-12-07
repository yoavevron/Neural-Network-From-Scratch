def cross_entropy(y_true, y_pred):  # Often used for classification
    """
    :param y_true: arrayyyyy
    :param y_pred: arrayyyyy
    :return:
    """
    print("1")
    #  item in y_true that represent the number 3 will be: [0,0,0,1,0,0,0,0,0,0]
    # Pred = np.arra(0, pred)
    # # the equivalent before softmax (logits) will be: [0.02, 0.1, 0.05, 0.84, 0, 0, 0 ,0, 0 ,0]
    #   epsilon = 1e-15
    # exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True)
    #   softmax_probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    # y_pred = np.clip(softmax_probs, epsilon, 1 - epsilon)
    # loss = -np.sum(y_true * np.log(y_pred))
    # num_samples = y_true.shape[0]
    # loss /= num_samples
    # return loss

def mean_squared_error():  # Often used  for regression
    return 1
