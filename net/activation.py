import numpy as np


def sigmoid(Z, deriv=False):
    if deriv:
        return sigmoid(Z) * (1 - sigmoid(Z))
    else:
        return 1 / (1 + np.exp(-Z))


def relu(Z, deriv=False):
    if deriv:
        return Z > 0
    else:
        return np.maximum(Z, 0)


def softmax(Z):
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)


def leaky_relu(Z, deriv=False):
    if deriv:
        return np.where(Z > 0, 1, 0.01)
    return np.where(Z > 0, Z, 0.01 * Z)
