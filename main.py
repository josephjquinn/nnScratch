from nn import nn
from data import DataProcessor

# fashion_data = "./data/fashion-mnist.csv"
numeric_data = "./data/numeric-mnist.csv"

numeric_labels = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

# fashion_labels = {
#     0: "T-shirt/top",
#     1: "Trouser",
#     2: "Pullover",
#     3: "Dress",
#     4: "Coat",
#     5: "Sandal",
#     6: "Shirt",
#     7: "Sneaker",
#     8: "Bag",
#     9: "Ankle boot",
# }


data_loader = DataProcessor(numeric_data)
data_loader.split_data()
data_loader.get_features_and_labels()

net = nn(
    input_nodes=784,
    hidden_nodes=50,
    output_nodes=10,
    act="relu",
    initialization="norm",
    labels=numeric_labels,
)


net.predict_grid(data_loader.x_test, 3)
test_accuracy = net.get_accuracy(data_loader.x_test, data_loader.y_test)
print(f"Test Accuracy before training: {test_accuracy * 100:.2f}%")

net.train(
    X_train=data_loader.x_train,
    Y_train=data_loader.y_train,
    X_dev=data_loader.x_dev,
    Y_dev=data_loader.y_dev,
    mini_batch=True,
    batch_size=1000,
    alpha=0.5,
    epochs=25,
    animate=True,
    plot=True,
    cmd=True,
)

test_accuracy = net.get_accuracy(data_loader.x_test, data_loader.y_test)
print(f"Test Accuracy after training: {test_accuracy * 100:.2f}%")
net.predict_grid(data_loader.x_test, 3)
