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
            param = row[0]
            acc = float(row[1])
            hyper.append([param, acc])
    return hyper


def plot_hyper(data, hyperparam):
    param = []
    acc = []
    print(data)
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
e_path = "./hyper/epoch.txt"
init_path = "./hyper/initialization.txt"
batch_path = "./hyper/batch.txt"


def optim_hl():
    hyper = []
    for i in range(10, 200, 10):
        net = nn(
            input_nodes=784,
            hidden_nodes=i,
            output_nodes=10,
            act="relu",
            initialization="rand",
            labels=None,
        )
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
    for i in np.arange(0.01, 1.0, 0.05):
        net = nn(
            input_nodes=784,
            hidden_nodes=10,
            output_nodes=10,
            act="relu",
            initialization="rand",
            labels=None,
        )
        net.train(
            processor.x_train,
            processor.y_train,
            processor.x_dev,
            processor.y_dev,
            alpha=i,
            epochs=100,
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
        net = nn(
            input_nodes=784,
            hidden_nodes=10,
            output_nodes=10,
            act=act_func,
            initialization="rand",
            labels=None,
        )
        net.train(
            processor.x_train,
            processor.y_train,
            processor.x_dev,
            processor.y_dev,
            alpha=0.01,
            epochs=100,
            animate=False,
            plot=False,
        )
        test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
        data = [act_func, test_accuracy]
        hyper.append(data)

    write_hyper(hyper, act_path)


def optim_e():
    hyper = []
    for i in range(100, 1000, 100):
        net = nn(
            input_nodes=784,
            hidden_nodes=10,
            output_nodes=10,
            act="relu",
            initialization="rand",
            labels=None,
        )
        net.train(
            processor.x_train,
            processor.y_train,
            processor.x_dev,
            processor.y_dev,
            alpha=0.01,
            epochs=i,
            animate=False,
            plot=False,
        )
        test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
        data = [i, test_accuracy]
        hyper.append(data)

    write_hyper(hyper, e_path)


def optim_init():
    hyper = []
    for init in ["norm", "rand", "He"]:
        net = nn(
            input_nodes=784,
            hidden_nodes=10,
            output_nodes=10,
            act="relu",
            initialization=init,
            labels=None,
        )
        net.train(
            processor.x_train,
            processor.y_train,
            processor.x_dev,
            processor.y_dev,
            alpha=0.01,
            epochs=100,
            animate=False,
            plot=False,
        )
        test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
        data = [init, test_accuracy]
        hyper.append(data)

    write_hyper(hyper, init_path)


def optim_batch():
    hyper = []
    net = nn(
        input_nodes=784,
        hidden_nodes=10,
        output_nodes=10,
        act="relu",
        initialization="rand",
        labels=None,
    )
    net.train(
        processor.x_train,
        processor.y_train,
        processor.x_dev,
        processor.y_dev,
        alpha=0.01,
        epochs=20,
        animate=False,
        plot=False,
        mini_batch=False,
        batch_size=None,
    )
    test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
    data = [0, test_accuracy]
    hyper.append(data)

    net = nn(
        input_nodes=784,
        hidden_nodes=10,
        output_nodes=10,
        act="relu",
        initialization="rand",
        labels=None,
    )
    net.train(
        processor.x_train,
        processor.y_train,
        processor.x_dev,
        processor.y_dev,
        alpha=0.01,
        epochs=20,
        animate=False,
        plot=False,
        mini_batch=True,
        batch_size=1,
    )
    test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
    data = [0, test_accuracy]
    hyper.append(data)
    for batch_size in range(1000, 10000, 1000):
        net = nn(
            input_nodes=784,
            hidden_nodes=10,
            output_nodes=10,
            act="relu",
            initialization="rand",
            labels=None,
        )
        net.train(
            processor.x_train,
            processor.y_train,
            processor.x_dev,
            processor.y_dev,
            alpha=0.01,
            epochs=20,
            animate=False,
            plot=False,
            mini_batch=True,
            batch_size=batch_size,
        )
        test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
        data = [batch_size, test_accuracy]
        hyper.append(data)

    write_hyper(hyper, batch_path)


processor = DataProcessor("./data/numeric-mnist.csv")
processor.split_data()
processor.get_features_and_labels()


# optim_hl()
# optim_lr()
# optim_act()
# optim_e()
# optim_init()
#optim_batch()

hyper_hl = read_hyper(hl_path)
plot_hyper(hyper_hl, "# of hidden layer neruons")

hyper_lr = read_hyper(hl_path)
plot_hyper(hyper_lr, "Learning Rate")

hyper_act = read_hyper(act_path)
plot_hyper(hyper_act, "Activation Function")
hyper_e = read_hyper(e_path)
plot_hyper(hyper_e, "Number of Epochs")


hyper_init = read_hyper(init_path)
plot_hyper(hyper_init, "Initialization Method")

hyper_batch = read_hyper(batch_path)
plot_hyper(hyper_batch, "Batch Size")
