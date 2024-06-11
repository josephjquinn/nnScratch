import numpy as np
from activation import ReLU, softmax, ReLU_deriv


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

    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, 10))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def backward_prop(self, Z1, A1, Z2, A2, X, Y):
        m = X.shape[1]
        one_hot_Y = self.one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = 1 / m * dZ2.dot(A1.T)
        db2 = 1 / m * np.sum(dZ2, 1)
        dZ1 = self.W2.T.dot(dZ2) * ReLU_deriv(Z1)
        dW1 = 1 / m * dZ1.dot(X.T)
        db1 = 1 / m * np.sum(dZ1, 1)
        return dW1, db1, dW2, db2

    def optimize(self, dW1, db1, dW2, db2, alpha):
        self.W1 -= alpha * dW1
        self.b1 -= alpha * np.reshape(db1, (10, 1))
        self.W2 -= alpha * dW2
        self.b2 -= alpha * np.reshape(db2, (10, 1))

    def train(self, X, Y, epochs, alpha):
        for i in range(epochs):
            Z1, A1, Z2, A2 = self.forward_prop(X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, X, Y)
            self.optimize(dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                accuracy = self.get_accuracy(X, Y)
                print(f"Epoch {i}, Accuracy: {accuracy:.4f}")

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
