from net.nn import nn
import numpy as np
from data import DataProcessor
import matplotlib.pyplot as plt


hl_path = "./hyper/hidden_layer.txt"
lr_hl_path = "./hyper/lr_hl_dual.txt"
act_init_path = "./hyper/act_init_dual.txt"
lr_path = "./hyper/learning_rate.txt"
act_path = "./hyper/activation.txt"
e_path = "./hyper/epoch.txt"
init_path = "./hyper/initialization.txt"
batch_path = "./hyper/batch.txt"


def flatten(xs):
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten(x))
    else:
        result.append(xs)
    return result


def write_hyper(data, path):
    np_data = np.array(data)
    np.savetxt(path, np_data, fmt="%f")


def plot_hyper(data, hyperparam, key=None):
    param = data[:, 0]
    param = np.round(param, decimals=3)
    if key:
        tmp = []
        for i in param:
            i = int(i)
            tmp.append(key[i])
        param = tmp
    acc = data[:, 1]
    plt.figure(figsize=(10, 6))
    plt.plot(param, acc, marker="o")
    plt.xlabel(hyperparam)
    plt.ylabel("Test Accuracy")
    plt.title(f"Test Accuracy vs {hyperparam}")
    plt.grid(True)
    plt.show()


def plot_hyper_loss(data, param, key=None):
    losses = data[:, 2:]
    labels = data[:, 0]
    epochs = losses[0].size
    plt.figure(figsize=(10, 6))
    for label, loss in zip(labels, losses):
        if key:
            label = int(label)
        plt.plot(range(epochs), loss, label=key[label] if key else label)
    plt.ylabel("Loss")
    plt.title("Validation Loss over Epochs")
    plt.legend(title=param)
    plt.grid(True)
    plt.show()


def plot_dual(data, title, xaxi, yaxi, x_key=None, y_key=None):
    data = np.array(data)
    z = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]

    unique_x = np.unique(x)
    unique_y = np.unique(y)

    if x_key:
        x_labels = [x_key[int(idx)] for idx in unique_x]
    else:
        x_labels = [f"{xi:.3f}" for xi in np.unique(x)]

    if y_key:
        y_labels = [y_key[int(idx)] for idx in unique_y]
    else:
        y_labels = [f"{yi:.3f}" for yi in np.unique(y)]

    Z = np.zeros((len(unique_y), len(unique_x)))
    for i in range(len(unique_x)):
        for j in range(len(unique_y)):
            mask = (x == unique_x[i]) & (y == unique_y[j])
            Z[j, i] = z[mask][0] if np.any(mask) else np.nan

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(Z, cmap="viridis", origin="lower")

    ax.set_xticks(np.arange(len(unique_x)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(unique_y)))
    ax.set_yticklabels(y_labels)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(len(unique_y)):
        for j in range(len(unique_x)):
            ax.text(
                j,
                i,
                f"{Z[i, j]:.3f}" if not np.isnan(Z[i, j]) else "",
                ha="center",
                va="center",
                color="w",
            )
    ax.set_title(title)
    ax.set_xlabel(xaxi)
    ax.set_ylabel(yaxi, rotation=-90, va="top")
    fig.tight_layout()
    cbar = plt.colorbar(im)
    cbar.set_label("Accuracy", rotation=-90, va="bottom")
    plt.show()


def optim_hl():
    hyper = []
    for i in range(10, 100, 10):
        net = nn(
            input_nodes=784,
            hidden_nodes=i,
            output_nodes=10,
            act="relu",
            initialization="norm",
            labels=None,
        )
        net.train(
            processor.x_train,
            processor.y_train,
            processor.x_dev,
            processor.y_dev,
            alpha=0.1,
            epochs=100,
            animate=False,
            plot=False,
        )
        test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
        data = [i, test_accuracy, net.val_losses]
        data = flatten(data)
        hyper.append(data)

    write_hyper(hyper, hl_path)


def optim_lr_hl():
    hyper = []
    for i in range(10, 60, 10):
        for j in np.arange(0.01, 0.1, 0.02):
            net = nn(
                input_nodes=784,
                hidden_nodes=i,
                output_nodes=10,
                act="relu",
                initialization="norm",
                labels=None,
            )
            net.train(
                processor.x_train,
                processor.y_train,
                processor.x_dev,
                processor.y_dev,
                alpha=j,
                epochs=100,
                animate=False,
                plot=False,
            )
            test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
            data = [test_accuracy, i, j]
            data = flatten(data)
            hyper.append(data)

    write_hyper(hyper, lr_hl_path)


def optim_act_init():
    hyper = []
    act_fns = ["sigmoid", "relu", "leaky"]
    init_methods = ["norm", "rand", "He"]
    for idx, fn in enumerate(act_fns):
        for jdx, init in enumerate(init_methods):
            net = nn(
                input_nodes=784,
                hidden_nodes=10,
                output_nodes=10,
                act=fn,
                initialization=init,
                labels=None,
            )
            net.train(
                processor.x_train,
                processor.y_train,
                processor.x_dev,
                processor.y_dev,
                alpha=0.1,
                epochs=100,
                animate=False,
                plot=False,
            )
            test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
            data = [test_accuracy, idx, jdx]
            data = flatten(data)
            hyper.append(data)

    write_hyper(hyper, act_init_path)


def optim_lr():
    hyper = []
    for i in np.arange(0.01, 1.0, 0.1):
        net = nn(
            input_nodes=784,
            hidden_nodes=10,
            output_nodes=10,
            act="relu",
            initialization="norm",
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
        data = [i, test_accuracy, net.val_losses]
        data = flatten(data)
        hyper.append(data)

    write_hyper(hyper, lr_path)


def optim_act():
    hyper = []
    act_fns = ["sigmoid", "relu", "leaky"]
    for idx, fn in enumerate(act_fns):
        net = nn(
            input_nodes=784,
            hidden_nodes=10,
            output_nodes=10,
            act=fn,
            initialization="He",
            labels=None,
        )
        net.train(
            processor.x_train,
            processor.y_train,
            processor.x_dev,
            processor.y_dev,
            alpha=0.1,
            epochs=100,
            animate=False,
            plot=False,
        )
        test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
        data = [idx, test_accuracy, net.val_losses]
        data = flatten(data)
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
    init_methods = ["norm", "rand", "He"]
    for idx, init in enumerate(init_methods):
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
            alpha=0.1,
            epochs=100,
            animate=False,
            plot=False,
        )
        test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
        data = [idx, test_accuracy, net.val_losses]
        data = flatten(data)
        hyper.append(data)

    write_hyper(hyper, init_path)


def optim_batch():
    hyper = []
    for batch_size in [10, 100, 1000, 10000, 30000]:
        net = nn(
            input_nodes=784,
            hidden_nodes=10,
            output_nodes=10,
            act="relu",
            initialization="rand",
            labels=None,
        )

        num_batches = processor.x_train.shape[1] // batch_size
        epoch_count = 500 // num_batches
        net.train(
            processor.x_train,
            processor.y_train,
            processor.x_dev,
            processor.y_dev,
            alpha=0.1,
            epochs=50,
            animate=False,
            plot=False,
            mini_batch=True,
            batch_size=batch_size,
        )
        test_accuracy = net.get_accuracy(processor.x_test, processor.y_test)
        data = [batch_size, test_accuracy, net.val_losses]
        data = flatten(data)
        hyper.append(data)

    write_hyper(hyper, batch_path)


processor = DataProcessor("./data/numeric-mnist.csv")
processor.split_data()
processor.get_features_and_labels()

init_methods = ["norm", "rand", "He"]
act_fns = ["sigmoid", "relu", "leaky"]

# optim_hl()
# optim_lr()
# optim_batch
# optim_init()
# optim_act()
# optim_act_init()
# optim_lr_hl()


data = np.loadtxt(lr_path)
plot_hyper(data, "Learning Rate")
plot_hyper_loss(data, "Learning Rate")

data = np.loadtxt(hl_path)
plot_hyper(data, "Hidden Layer Neurons Count")
plot_hyper_loss(data, "Hidden Layer Neurons Count")


data = np.loadtxt(lr_hl_path)
plot_dual(
    data,
    "Learning Rate vs Hidden Layer Neuron Count",
    "Hidden Layer Neuron Count",
    "Learning Rate",
)

data = np.loadtxt(act_path)
plot_hyper(data, "Activation Function", key=act_fns)
plot_hyper_loss(data, "Activation Function", key=act_fns)

data = np.loadtxt(init_path)
plot_hyper(data, "Initialization Method", key=init_methods)
plot_hyper_loss(data, "Initialization Method", key=init_methods)

data = np.loadtxt(act_init_path)
plot_dual(
    data,
    "Activation Function vs Initialization Method",
    "Activation Function",
    "Initialization Method",
    act_fns,
    init_methods,
)

data = np.loadtxt(batch_path)
plot_hyper(data, "Batch Size")
plot_hyper_loss(data, "Batch Size")
