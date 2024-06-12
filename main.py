from nn import nn
from data import DataProcessor
from matplotlib import pyplot as plt

processor = DataProcessor("./MNIST.csv")
processor.split_data()
processor.get_features_and_labels()
processor.print_shapes()

net = nn()


# def test_prediction(index):
#     current_image = processor.x_train[:, index, None]
#     prediction = net.predict(current_image)
#     label = processor.y_train[index]
#     print("Prediction: ", prediction)
#     print("Label: ", label)
#
#     current_image = current_image.reshape((28, 28)) * 255
#     plt.gray()
#     plt.imshow(current_image, interpolation="nearest")
#     plt.show()


# net.predict(processor.x_train, 45)

train_accuracy = net.get_accuracy(processor.x_train, processor.y_train)
test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

net.train(processor.x_train, processor.y_train, processor.x_dev, processor.y_dev, 100, 0.1)

train_accuracy = net.get_accuracy(processor.x_train, processor.y_train)
test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)

print(f"Training Accuracy after training: {train_accuracy * 100:.2f}%")
print(f"Test Accuracy after training: {test_accuracy * 100:.2f}%")

# test_prediction(8)
