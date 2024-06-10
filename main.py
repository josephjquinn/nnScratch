from nn import nn
from data import DataProcessor
from matplotlib import pyplot as plt
import numpy as np


processor = DataProcessor("./train.csv")
processor.split_data()
processor.get_features_and_labels()
processor.print_shapes()


def make_predictions(X):
    _, _, _, A2 = net.forward_prop(X)
    predictions = get_predictions(A2)
    return predictions


def get_predictions(A2):
    return np.argmax(A2, axis=0)


def test_prediction(index):
    current_image = processor.x_train[:, index, None]
    prediction = net.predict(current_image)
    label = processor.y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation="nearest")
    plt.show()


net = nn()
test_prediction(8)
