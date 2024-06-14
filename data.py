import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class DataProcessor:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.data = np.array(self.data)
        np.random.shuffle(self.data)
        self.m, self.n = self.data.shape
        self.v_sec = int(self.m * 0.1)

    def split_data(self):
        self.dev = self.data[0 : self.v_sec]
        self.test = self.data[self.v_sec : 2 * self.v_sec]
        self.train = self.data[2 * self.v_sec :]

    def get_features_and_labels(self):
        self.y_dev = self.dev[:, 0].T
        self.x_dev = self.dev[:, 1:].T
        self.x_dev = self.x_dev / 255

        self.y_test = self.test[:, 0].T
        self.x_test = self.test[:, 1:].T
        self.x_test = self.x_test / 255

        self.y_train = self.train[:, 0].T
        self.x_train = self.train[:, 1:].T
        self.x_train = self.x_train / 255

    def print_shapes(self):
        print("Y Train", self.y_train.shape)
        print("X Train", self.x_train.shape)
        print("Y Dev", self.y_dev.shape)
        print("X Dev", self.x_dev.shape)
        print("Y Test", self.y_test.shape)
        print("X Test", self.x_test.shape)

    def visualize_training_images(self):
        for i in range(self.x_train.shape[1]):
            x = self.x_train[:, i].reshape((28, 28))
            plt.gray()
            plt.imshow(x, interpolation="nearest")
            plt.title(f"Label: {self.y_train[i]}")
            plt.draw()
            plt.pause(0.01)
            plt.clf()


if __name__ == "__main__":
    processor = DataProcessor("./MNIST.csv")
    processor.split_data()
    processor.get_features_and_labels()
    processor.print_shapes()
    processor.visualize_training_images()
