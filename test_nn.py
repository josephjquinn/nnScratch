import numpy as np
import pytest
from nn import nn


@pytest.fixture
def nn_instance():
    return nn()


def test_initialize_parameters(nn_instance):
    assert nn_instance.W1.shape == (10, 784)
    assert nn_instance.W2.shape == (10, 10)
    assert nn_instance.B1.shape == (10,)
    assert nn_instance.B2.shape == (10,)
    assert np.all(nn_instance.W1 >= -0.5) and np.all(nn_instance.W1 <= 0.5)
    assert np.all(nn_instance.W2 >= -0.5) and np.all(nn_instance.W2 <= 0.5)
    assert np.all(nn_instance.B1 >= -0.5) and np.all(nn_instance.B1 <= 0.5)
    assert np.all(nn_instance.B2 >= -0.5) and np.all(nn_instance.B2 <= 0.5)


def test_forward_prop(nn_instance):
    X = np.random.rand(784)
    Z1, A1, Z2, A2 = nn_instance.forward_prop(X)

    assert Z1.shape == (10,)
    assert A1.shape == (10,)
    assert Z2.shape == (10,)
    assert A2.shape == (10,)

    assert np.all(A1 >= 0)

    assert np.isclose(np.sum(A2), 1)
