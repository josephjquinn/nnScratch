import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def ReLU(Z):
    return np.maximum(Z, 0)


def ReLU_deriv(z):
    return z > 0


def softmax(Z):
    exp = np.exp(Z - np.max(Z))
    return exp / exp.sum(axis=0)
