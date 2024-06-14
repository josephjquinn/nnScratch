from nn import nn
import numpy as np
from data import DataProcessor
import matplotlib.pyplot as plt
import csv


def write_hyper(data, path):
    with open(path, "w") as f:
        csv.writer(f, delimiter=" ").writerows(data)


def read_hyper(path):
    hyper = []
    with open(path, "r") as f:
        reader = csv.reader(f, delimiter=" ")
        for row in reader:
            param = float(row[0])
            acc = float(row[1])
            hyper.append([param, acc])
    return hyper


def plot_hyper(data, hyperparam):
    param = []
    acc = []
    for i in data:
        param.append(i[0])
        acc.append(i[1])

    plt.ylabel("Accuracy")
    plt.close("all")
    plt.clf()
    plt.cla()
    plt.plot(param, acc)
    plt.xlabel(hyperparam)
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.show()


hl_path = "./hyper/hidden_layer_neruons.txt"
lr_path = "./hyper/learning_rate.txt"
act_path = "./hyper/activation.txt"


def optim_hl():
    hyper = []
    for i in range(10, 160, 10):
        net = nn(i + 1, "relu", "He")
        net.train(
            processor.x_train,
            processor.y_train,
            processor.x_dev,
            processor.y_dev,
            alpha=0.01,
            epochs=50,
            animate=False,
            plot=False,
        )

        test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
        data = [i, test_accuracy]
        hyper.append(data)

    write_hyper(hyper, hl_path)


def optim_lr():
    hyper = []
    for i in np.arange(0.01, 0.2, 0.01):
        net = nn(10, "relu", "He")
        net.train(
            processor.x_train,
            processor.y_train,
            processor.x_dev,
            processor.y_dev,
            alpha=i,
            epochs=500,
            animate=False,
            plot=False,
        )

        test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
        data = [i, test_accuracy]
        hyper.append(data)

    write_hyper(hyper, lr_path)


def optim_act():
    hyper = []
    for act_func in ["sigmoid", "relu", "leaky"]:
        net = nn(10, 10, 10, act_func, "He")
        net.train(
            processor.x_train,
            processor.y_train,
            processor.x_dev,
            processor.y_dev,
            alpha=0.01,
            epochs=500,
            animate=False,
            plot=False,
        )

        test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
        data = [act_func, test_accuracy]
        hyper.append(data)

    write_hyper(hyper, act_path)


processor = DataProcessor("./data/numeric-mnist.csv.csv")
processor.split_data()
processor.get_features_and_labels()
processor.print_shapes()


# optim_hl()
hyper_hl = read_hyper(hl_path)
plot_hyper(hyper_hl, "# of hidden layer neruons")

# optim_lr()
hyper_lr = read_hyper(hl_path)
plot_hyper(hyper_lr, "Learning Rate")
