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

# def cross_entropy(A2, Y):
#     m = Y.size  
#     log_likelihood = -np.log(A2[Y, np.arange(m)])
#     loss = np.sum(log_likelihood) / m
#     return loss


def cross_entropy(A2, Y):
    m = Y.size  
    log_likelihood = -np.log(A2[Y, np.arange(m)])
    loss = np.sum(log_likelihood) / m
    return loss
