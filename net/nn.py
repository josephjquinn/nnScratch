from net.loss import cross_entropy
import numpy as np
from net.activation import relu, sigmoid, softmax, leaky_relu
from matplotlib import pyplot as plt
import json


class nn:
    def __init__(
        self,
        input_nodes,
        hidden_nodes,
        output_nodes,
        act,
        initialization="rand",
        labels=None,
    ):
        self.hidden_nodes = hidden_nodes
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.labels = (
            labels if labels is not None else {i: str(i) for i in range(output_nodes)}
        )
        self.initialization = initialization
        self.initialize_parameters()
        if act == "sigmoid":
            self.act = sigmoid
        elif act == "relu":
            self.act = relu
        elif act == "leaky":
            self.act = leaky_relu
        else:
            raise ValueError("Invalid activation function")

    def initialize_parameters(self):
        if self.initialization == "norm":
            self.W1 = np.random.normal(
                size=(self.hidden_nodes, self.input_nodes)
            ) * np.sqrt(1.0 / self.input_nodes)
            self.b1 = np.random.normal(size=(self.hidden_nodes, 1)) * np.sqrt(
                1.0 / self.hidden_nodes
            )
            self.W2 = np.random.normal(
                size=(self.output_nodes, self.hidden_nodes)
            ) * np.sqrt(1.0 / self.hidden_nodes)
            self.b2 = np.random.normal(size=(self.output_nodes, 1)) * np.sqrt(
                1.0 / self.hidden_nodes
            )
        elif self.initialization == "He":
            self.W1 = np.random.normal(
                size=(self.hidden_nodes, self.input_nodes)
            ) * np.sqrt(2.0 / self.input_nodes)
            self.b1 = np.zeros((self.hidden_nodes, 1))
            self.W2 = np.random.normal(
                size=(self.output_nodes, self.hidden_nodes)
            ) * np.sqrt(2.0 / self.hidden_nodes)
            self.b2 = np.zeros((self.output_nodes, 1))
        elif self.initialization == "rand":
            self.W1 = np.random.rand(self.hidden_nodes, self.input_nodes) - 0.5
            self.b1 = np.random.rand(self.hidden_nodes, 1) - 0.5
            self.W2 = np.random.rand(self.output_nodes, self.hidden_nodes) - 0.5
            self.b2 = np.random.rand(self.output_nodes, 1) - 0.5
        else:
            raise ValueError("Invalid initialization function")

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = self.act(Z1)
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
        dZ1 = dA1 * self.act(Z1, deriv=True)
        dW1 = dZ1.dot(X.T)
        db1 = np.sum(dZ1, axis=1, keepdims=True)

        return dW1, db1, dW2, db2

    def optimize(self, dW1, db1, dW2, db2, alpha):
        self.W1 -= alpha * dW1
        self.b1 -= alpha * db1
        self.W2 -= alpha * dW2
        self.b2 -= alpha * db2

    def train(
        self,
        X_train,
        Y_train,
        X_dev,
        Y_dev,
        epochs,
        alpha,
        mini_batch=False,
        batch_size=None,
        animate=False,
        plot=True,
        cmd=True,
        save=False,
    ):
        self.train_losses = []
        self.val_losses = []
        self.accuracies = []
        self.batch_loss = []
        if plot or animate:
            plt.ion()
            plt.figure(figsize=(15, 6))

        if mini_batch:
            num_batches = X_train.shape[1] // batch_size

        for i in range(epochs):
            if mini_batch:
                cur_batch_loss = []
                for j in range(num_batches):
                    start = j * batch_size
                    end = (j + 1) * batch_size
                    X_batch = X_train[:, start:end]
                    Y_batch = Y_train[start:end]

                    Z1, A1, Z2, A2 = self.forward_prop(X_batch)
                    train_loss = cross_entropy(A2, Y_batch)
                    cur_batch_loss.append(train_loss)
                    self.batch_loss.append(train_loss)
                    dW1, db1, dW2, db2 = self.backward_prop(
                        Z1, A1, Z2, A2, X_batch, Y_batch
                    )
                    self.optimize(dW1, db1, dW2, db2, alpha)
                    self.print_batch_result(j + 1, num_batches, train_loss)
                self.train_losses.append(sum(cur_batch_loss) / num_batches)
            else:
                Z1, A1, Z2, A2 = self.forward_prop(X_train)
                train_loss = cross_entropy(A2, Y_train)
                self.train_losses.append(train_loss)
                dW1, db1, dW2, db2 = self.backward_prop(
                    Z1, A1, Z2, A2, X_train, Y_train
                )
                self.optimize(dW1, db1, dW2, db2, alpha)
            _, _, _, val_pred = self.forward_prop(X_dev)
            val_loss = cross_entropy(val_pred, Y_dev)
            accuracy = self.get_accuracy(X_dev, Y_dev)

            self.val_losses.append(val_loss)
            self.accuracies.append(accuracy)

            if animate:
                self.plot(mini_batch)

            if cmd:
                if i % 10 == 0:
                    self.print_epoch_result(i, accuracy, train_loss, val_loss)

        plt.ioff()
        if plot:
            plt.show()
        if save:
            self.save_training_data(save)

    def print_epoch_result(self, epoch, acc, train_loss, val_loss):
        print("_________________________")
        print(f"Epoch {epoch + 1}, Accuracy: {acc:.4f}")
        print(f"Train Loss {train_loss}, Val Loss {val_loss}")
        print("_________________________")

    def print_batch_result(self, batch, num_batches, train_loss):
        print(f"Batch {batch}/{num_batches}, Train Loss: {train_loss}")

    def plot(self, mb):
        rows = 1
        if mb:
            rows = 2
        plt.clf()
        plt.subplot(rows, 2, 1)
        plt.title("Epoch Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(self.train_losses, "-r", label="train")
        plt.plot(self.val_losses, "-b", label="val")
        plt.legend()
        plt.ylim(ymin=0)
        plt.subplot(rows, 2, 2)
        plt.title("Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.plot(self.accuracies)
        plt.ylim(0, 1)
        if mb:
            plt.subplot(2, 2, (3, 4))
            plt.plot(self.batch_loss, linewidth=0.2)
            plt.title("Step Loss")
            plt.xlabel("Step")
            plt.ylabel("Loss")
            plt.subplots_adjust(hspace=0.4)
        plt.show()
        plt.pause(0.1)

    def predict_single(self, X, index):
        current_image = X[:, index, None]
        _, _, _, A2 = self.forward_prop(current_image)
        pred = np.argmax(A2)
        current_image = current_image.reshape((28, 28)) * 255
        plt.gray()
        plt.imshow(current_image, interpolation="nearest")
        pred_label = self.labels[pred]
        plt.title(f"Pred: {pred_label}", fontsize=15)
        plt.show()

    def predict_grid(self, X, grid_size):
        grid_predictions = np.zeros((grid_size, grid_size), dtype=int)
        grid_images = np.zeros((grid_size, grid_size, 28, 28))

        plt.figure(figsize=(9, 9))

        for i in range(grid_size):
            for j in range(grid_size):
                index = np.random.randint(X.shape[1])
                current_image = X[:, index, None]
                _, _, _, A2 = self.forward_prop(current_image)
                pred = np.argmax(A2)
                grid_predictions[i, j] = pred
                grid_images[i, j] = current_image.reshape((28, 28)) * 255
                plt.subplot(grid_size, grid_size, i * grid_size + j + 1)
                plt.imshow(grid_images[i, j], cmap="gray")
                pred_label = self.labels[pred]
                plt.title(f"Pred: {pred_label}", fontsize=15)
                plt.axis("off")

        plt.show()

    def get_accuracy(self, X, Y):
        _, _, _, A2 = self.forward_prop(X)
        pred = np.argmax(A2, axis=0)
        accuracy = np.mean(pred == Y)
        return accuracy

    def save_model(self, path):
        model_params = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "labels": self.labels,
            "act": self.act.__name__,
            "input_nodes": self.input_nodes,
            "hidden_nodes": self.hidden_nodes,
            "output_nodes": self.output_nodes,
            "initialization": self.initialization,
        }
        np.savez(path, **model_params)
        print(f"Model saved to {path}")

    def load_model(self, path):
        loaded_params = np.load(path, allow_pickle=True)
        self.W1 = loaded_params["W1"]
        self.b1 = loaded_params["b1"]
        self.W2 = loaded_params["W2"]
        self.b2 = loaded_params["b2"]
        self.labels = loaded_params["labels"].item()
        activation_name = loaded_params["act"]
        print(activation_name)
        self.input_nodes = loaded_params["input_nodes"]
        self.hidden_nodes = loaded_params["hidden_nodes"]
        self.output_nodes = loaded_params["output_nodes"]
        self.initialization = loaded_params["initialization"]

        if activation_name == "sigmoid":
            self.act = sigmoid
        elif activation_name == "relu":
            self.act = relu
        elif activation_name == "leaky":
            self.act = leaky_relu
        else:
            raise ValueError("Invalid activation function name loaded from file")

        print(f"Model loaded from {path}")

    def save_training_data(self, path):
        train_data = {
            "Train_Loss": self.train_losses,
            "Val_Loss": self.val_losses,
            "Step_Loss": self.batch_loss,
            "Accuracy": self.accuracies,
        }

        with open(path, "w") as json_file:
            json.dump(train_data, json_file)

        print(f"Training Data saved to {path}")
