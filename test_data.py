import pytest
from matplotlib import pyplot as plt
from data import DataProcessor


@pytest.fixture(scope="module")
def data_processor():
    processor = DataProcessor("./MNIST.csv")
    processor.split_data()
    processor.get_features_and_labels()
    return processor


def test_data_loading(data_processor):
    assert data_processor.data.shape == (42000, 785)
    assert data_processor.m == 42000
    assert data_processor.n == 785


def test_split_data(data_processor):
    assert data_processor.val.shape == ((data_processor.m * 0.10), 785)
    assert data_processor.test.shape == ((data_processor.m * 0.10), 785)
    assert data_processor.train.shape == ((data_processor.m * 0.80), 785)


def test_get_features_and_labels(data_processor):
    assert data_processor.y_val.shape == (4200,)
    assert data_processor.x_val.shape == (784, 4200)
    assert data_processor.y_test.shape == (4200,)
    assert data_processor.x_test.shape == (784, 4200)
    assert data_processor.y_train.shape == (33600,)
    assert data_processor.x_train.shape == (784, 33600)


def test_visualize_training_images(data_processor, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(plt, "pause", lambda x: None)
    monkeypatch.setattr(plt, "clf", lambda: None)

    try:
        data_processor.visualize_training_images()
    except Exception as e:
        pytest.fail(f"visualize_training_images raised an exception: {e}")
