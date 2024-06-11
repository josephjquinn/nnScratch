import numpy as np
from activation import ReLU
from activation import softmax


class nn:
    def __init__(self):
        self.initialize_parameters()

    def initialize_parameters(self):
        self.W1 = np.random.rand(10, 784) - 0.5
        self.W2 = np.random.rand(10, 10) - 0.5
        self.b1 = np.random.rand(10, 1) - 0.5
        self.b2 = np.random.rand(10, 1) - 0.5
        self.learning_rate = 0.01

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(self, Z1, A1, Z2, A2, W1, W2, X, Y):
        m = X.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = np.dot(W2.T, dZ2) * (A1 > 0)
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def optimize(self, dW1, db1, dW2, db2):
        self.W1 = self.W1 - self.alpha * dW1
        self.b1 = self.b1 - self.alpha * db1
        self.W2 = self.W2 - self.alpha * dW2
        self.b2 = self.b2 - self.alpha * db2

    def train(self, X, val, epochs):
        pass

    def predict(self, X):
        _, _, _, A2 = self.forward_prop(X)
        pred = np.argmax(A2, axis=0)
        return pred

    def get_predictions(self, A2):
        return np.argmax(A2, axis=0)

    def make_predictions(self, X):
        _, _, _, A2 = self.forward_prop(X)
        predictions = self.get_predictions(A2)
        return predictions

    def get_accuracy(self, X, Y):
        predictions = self.make_predictions(X)
        accuracy = np.mean(predictions == Y)
        return accuracy
