from loss import cross_entropy
import numpy as np
from activation import ReLU, softmax, ReLU_deriv
from matplotlib import pyplot as plt


class nn:
    # def __init__(self):
    #     self.initialize_parameters()
    #
    # def initialize_parameters(self):
    #     # self.W1 = np.random.rand(10, 784) - 0.5
    #     # self.W2 = np.random.rand(10, 10) - 0.5
    #     # self.b1 = np.random.rand(10, 1) - 0.5
    #     # self.b2 = np.random.rand(10, 1) - 0.5
    #     #
    #     self.W1 = np.random.normal(size=(10, 784)) * np.sqrt(1.0 / (784))
    #     self.b1 = np.random.normal(size=(10, 1)) * np.sqrt(1.0 / 10)
    #     self.W2 = np.random.normal(size=(10, 10)) * np.sqrt(1.0 / 20)
    #     self.b2 = np.random.normal(size=(10, 1)) * np.sqrt(1.0 / (784))

    def __init__(self, hidden_nodes, act, normalization):
        self.hidden_nodes = hidden_nodes
        self.initialize_parameters()

    def initialize_parameters(self):
        input_nodes = 784
        output_nodes = 10
        self.W1 = np.random.normal(size=(self.hidden_nodes, input_nodes)) * np.sqrt(
            1.0 / input_nodes
        )
        self.b1 = np.random.normal(size=(self.hidden_nodes, 1)) * np.sqrt(
            1.0 / self.hidden_nodes
        )
        self.W2 = np.random.normal(size=(output_nodes, self.hidden_nodes)) * np.sqrt(
            1.0 / self.hidden_nodes
        )
        self.b2 = np.random.normal(size=(output_nodes, 1)) * np.sqrt(
            1.0 / self.hidden_nodes
        )

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

    def train(self, X_train, Y_train, X_dev, Y_dev, epochs, alpha):
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        for i in range(epochs):
            Z1, A1, Z2, A2 = self.forward_prop(X_train)
            train_loss = cross_entropy(A2, Y_train)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, X_train, Y_train)
            self.optimize(dW1, db1, dW2, db2, alpha)

            _, _, _, val_pred = self.forward_prop(X_dev)

            val_loss = cross_entropy(val_pred, Y_dev)
            accuracy = self.get_accuracy(X_dev, Y_dev)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.accuracies.append(accuracy)
            self.plot(persist=False)
            if i % 10 == 0:
                print(f"Epoch {i}, Accuracy: {accuracy:.4f}")
                print(f"Train Loss {train_loss}, Val Loss {val_loss}")
        self.plot(persist=True)

    def train_mini_batch(
        self, X_train, Y_train, X_dev, Y_dev, epochs, alpha, batch_size
    ):
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        num_batches = X_train.shape[1] // batch_size

        for i in range(epochs):
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size
                X_batch = X_train[:, start:end]
                Y_batch = Y_train[start:end]

                Z1, A1, Z2, A2 = self.forward_prop(X_batch)
                train_loss = cross_entropy(A2, Y_batch)
                dW1, db1, dW2, db2 = self.backward_prop(
                    Z1, A1, Z2, A2, X_batch, Y_batch
                )
                self.optimize(dW1, db1, dW2, db2, alpha)

            _, _, _, val_pred = self.forward_prop(X_dev)
            val_loss = cross_entropy(val_pred, Y_dev)
            accuracy = self.get_accuracy(X_dev, Y_dev)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.accuracies.append(accuracy)
            self.plot(persist=False)

            if i % 10 == 0:
                print(f"Epoch {i}, Accuracy: {accuracy:.4f}")
                print(f"Train Loss {train_loss}, Val Loss {val_loss}")

        self.plot(persist=True)

    def plot(self, persist):
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(self.train_losses, "-r", label="train")
        plt.plot(self.val_losses, "-b", label="val")
        plt.legend()
        plt.ylim(ymin=0)
        plt.subplot(1, 2, 2)
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(self.accuracies)
        plt.ylim(0, 1)
        plt.show(block=False)
        plt.pause(0.01)
        if persist:
            plt.show()

    def predict(self, X, index):
        current_image = X[:, index, None]
        _, _, _, A2 = self.forward_prop(current_image)
        pred = np.argmax(A2)
        print("Prediction: ", pred)
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation="nearest")
        plt.show()

    def get_accuracy(self, X, Y):
        _, _, _, A2 = self.forward_prop(X)
        pred = np.argmax(A2, axis=0)
        accuracy = np.mean(pred == Y)
        return accuracy
