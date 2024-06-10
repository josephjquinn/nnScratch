import numpy as np
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def ReLU(z):
    return np.maximum(z, 0)


def softmax(z):
    A = np.exp(z) / sum(np.exp(z))
    return A
