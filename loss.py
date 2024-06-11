import numpy as np
from activation import softmax


def MSE(act, pred):
    diff = pred - act
    differences_squared = diff**2
    loss = differences_squared.mean()

    return loss


def RMSE(act, pred):
    diff = pred - act
    differences_squared = diff**2
    mean_diff = differences_squared.mean()
    loss = np.sqrt(mean_diff)
    return loss


def MAE(act, pred):
    diff = pred - act
    abs_diff = np.absolute(diff)
    loss = abs_diff.mean()
    return loss


def cross_entropy(y_pred, y_true):
    y_pred = softmax(y_pred)
    loss = 0

    for i in range(len(y_pred)):
        loss = loss + (-1 * y_true[i] * np.log(y_pred[i]))

    return loss
