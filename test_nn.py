import numpy as np
import pytest
from nn import nn


@pytest.fixture(scope="module")
def neural_network():
    return nn()


def test_initialize_parameters(neural_network):
    assert neural_network.W1.shape == (10, 784)
    assert neural_network.W2.shape == (10, 10)
    assert neural_network.b1.shape == (10, 1)
    assert neural_network.b2.shape == (10, 1)


def test_forward_propagation(neural_network):
    X = np.random.rand(784, 1)

    Z1, A1, Z2, A2 = neural_network.forward_prop(X)

    assert Z1.shape == (10, 1)
    assert A1.shape == (10, 1)
    assert Z2.shape == (10, 1)
    assert A2.shape == (10, 1)


def test_back_prop(neural_network):
    X = np.random.rand(784, 1)
    Y = np.random.rand(10, 1)

    Z1, A1, Z2, A2 = neural_network.forward_prop(X)

    dW1, db1, dW2, db2 = neural_network.backward_prop(
        Z1, A1, Z2, A2, neural_network.W1, neural_network.W2, X, Y
    )

    assert dW1.shape == (10, 784)
    assert db1.shape == (10, 1)
    assert dW2.shape == (10, 10)
    assert db2.shape == (10, 1)


def test_one_hot(neural_network):
    Y = np.random.randint(low=0, high=10, size=125)
    one_hot_y = neural_network.one_hot(Y)
    print(one_hot_y.shape)
    assert one_hot_y.shape == (10, 125)


def test_predict(neural_network):
    X = np.random.rand(784, 1)
    pred = neural_network.predict(X)
    assert pred.shape == (1,)
