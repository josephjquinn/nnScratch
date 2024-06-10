import numpy as np
from act import ReLU
from act import softmax


class nn:
    def __init__(self):
        self.initialize_parameters()

    def initialize_parameters(self):
        self.W1 = np.random.rand(10, 784) - 0.5
        self.W2 = np.random.rand(10, 10) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    def predict(self, X):
        _, _, _, A2 = self.forward_prop(X)
        predictions = np.argmax(A2, axis=0)
        return predictions

    def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
        pass


net = nn()
print(net.W1)
print(net.W1.shape)
print(net.W1.shape)
