from nn import nn
from data import DataProcessor

processor = DataProcessor("./MNIST.csv")
processor.split_data()
processor.get_features_and_labels()
processor.print_shapes()

net = nn(10, "relu", "He")


# net.predict(processor.x_train, 45)
# net.predict_grid(processor.x_train, 3)
test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

net.train(
    processor.x_train,
    processor.y_train,
    processor.x_dev,
    processor.y_dev,
    # mini_batch=True,
    alpha=0.06,
    # batch_size=10,
    epochs=20,
    animate=True,
)

test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)

print(f"Test Accuracy after training: {test_accuracy * 100:.2f}%")

# net.predict_grid(processor.x_train, 3)
# net.predict(processor.x_train, 45)
