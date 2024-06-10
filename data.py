import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class DataProcessor:
    def __init__(self, filepath):
        self.data = pd.read_csv(filepath)
        self.data = np.array(self.data)
        self.m, self.n = self.data.shape
        self.v_sec = int(self.m * 0.1)

    def split_data(self):
        self.val = self.data[0 : self.v_sec]
        self.test = self.data[self.v_sec : 2 * self.v_sec]
        self.train = self.data[2 * self.v_sec :]

    def get_features_and_labels(self):
        self.val_y = self.val[:, 0].T
        self.val_x = self.val[:, 1:].T

        self.y_test = self.test[:, 0].T
        self.x_test = self.test[:, 1:].T

        self.y_train = self.train[:, 0].T
        self.x_train = self.train[:, 1:].T

    def print_shapes(self):
        print(self.y_train.shape)
        print(self.x_train.shape)
        print(self.y_train)
        print(self.x_train)

    def visualize_training_images(self):
        for i, (x, y) in enumerate(zip(self.x_train, self.y_train)):
            x = x.reshape((28, 28)) * 255
            plt.gray()
            plt.imshow(x, interpolation="nearest")
            plt.title(f"Label: {y}")
            plt.draw()
            plt.pause(0.01)
            plt.clf()


if __name__ == "__main__":
    processor = DataProcessor("./train.csv")
    processor.split_data()
    processor.get_features_and_labels()
    processor.visualize_training_images()
