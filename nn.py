from loss import cross_entropy
import numpy as np
from activation import ReLU, softmax, ReLU_deriv
from matplotlib import pyplot as plt


class nn:
    def __init__(self):
        self.initialize_parameters()

    def initialize_parameters(self):
        # self.W1 = np.random.rand(10, 784) - 0.5
        # self.W2 = np.random.rand(10, 10) - 0.5
        # self.b1 = np.random.rand(10, 1) - 0.5
        # self.b2 = np.random.rand(10, 1) - 0.5
        #
        self.W1 = np.random.normal(size=(10, 784)) * np.sqrt(1.0 / (784))
        self.b1 = np.random.normal(size=(10, 1)) * np.sqrt(1.0 / 10)
        self.W2 = np.random.normal(size=(10, 10)) * np.sqrt(1.0 / 20)
        self.b2 = np.random.normal(size=(10, 1)) * np.sqrt(1.0 / (784))

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
        dZ2 = (A2 - self.one_hot(Y)) / Y.size
        dW2 = dZ2.dot(A1.T)
        db2 = np.sum(dZ2, axis=1, keepdims=True)
        dA1 = self.W2.T.dot(dZ2)
        dZ1 = dA1 * ReLU_deriv(Z1)
        dW1 = dZ1.dot(X.T)
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def optimize(self, dW1, db1, dW2, db2, alpha):
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2

    def train(self, X, Y, epochs, alpha):
        self.losses = []
        self.accuracies = []
        for i in range(epochs):
            Z1, A1, Z2, A2 = self.forward_prop(X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, X, Y)
            self.optimize(dW1, db1, dW2, db2, alpha)
            self.plot()
            if i % 10 == 0:
                accuracy = self.get_accuracy(X, Y)
                loss = cross_entropy(A2, Y)
                self.losses.append(loss)
                self.accuracies.append(accuracy)
                print(f"Epoch {i}, Accuracy: {accuracy:.4f}")
                print(f"Loss {loss}")
                print(loss)

    def plot(self):
        plt.clf()
        plt.title("Training...")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(self.accuracies)
        plt.plot(self.losses, "-r", label="train")
        plt.legend()
        plt.ylim(ymin=0)
        plt.show(block=False)
        plt.pause(0.01)

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
