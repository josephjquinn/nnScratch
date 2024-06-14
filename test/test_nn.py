import pytest
import numpy as np
from nn import nn


def create_synthetic_data(num_samples, num_features, num_classes):
    X = np.random.randn(num_features, num_samples)
    Y = np.random.randint(0, num_classes, num_samples)
    return X, Y


def test_initialization():
    input_nodes = 784
    hidden_nodes = 64
    output_nodes = 10
    activation = "relu"
    initialization = "He"

    net = nn(input_nodes, hidden_nodes, output_nodes, activation, initialization)
    assert net.W1.shape == (hidden_nodes, input_nodes)
    assert net.b1.shape == (hidden_nodes, 1)
    assert net.W2.shape == (output_nodes, hidden_nodes)
    assert net.b2.shape == (output_nodes, 1)


def test_forward_propagation():
    input_nodes = 784
    hidden_nodes = 64
    output_nodes = 10
    activation = "relu"
    initialization = "He"

    net = nn(input_nodes, hidden_nodes, output_nodes, activation, initialization)
    X, _ = create_synthetic_data(100, input_nodes, output_nodes)
    Z1, A1, Z2, A2 = net.forward_prop(X)

    assert Z1.shape == (hidden_nodes, X.shape[1])
    assert A1.shape == (hidden_nodes, X.shape[1])
    assert Z2.shape == (output_nodes, X.shape[1])
    assert A2.shape == (output_nodes, X.shape[1])


def test_backward_propagation():
    input_nodes = 784
    hidden_nodes = 64
    output_nodes = 10
    activation = "relu"
    initialization = "He"

    net = nn(input_nodes, hidden_nodes, output_nodes, activation, initialization)
    X, Y = create_synthetic_data(100, input_nodes, output_nodes)
    Z1, A1, Z2, A2 = net.forward_prop(X)
    dW1, db1, dW2, db2 = net.backward_prop(Z1, A1, Z2, A2, X, Y)

    assert dW1.shape == net.W1.shape
    assert db1.shape == net.b1.shape
    assert dW2.shape == net.W2.shape
    assert db2.shape == net.b2.shape


def test_training():
    input_nodes = 784
    hidden_nodes = 64
    output_nodes = 10
    activation = "relu"
    initialization = "He"
    epochs = 1
    alpha = 0.01

    net = nn(input_nodes, hidden_nodes, output_nodes, activation, initialization)
    X_train, Y_train = create_synthetic_data(100, input_nodes, output_nodes)
    X_dev, Y_dev = create_synthetic_data(20, input_nodes, output_nodes)

    net.train(
        X_train,
        Y_train,
        X_dev,
        Y_dev,
        epochs,
        alpha,
        mini_batch=False,
        plot=False,
        cmd=False,
    )

    assert len(net.train_losses) == epochs
    assert len(net.val_losses) == epochs
    assert len(net.accuracies) == epochs




if __name__ == "__main__":
    pytest.main()
