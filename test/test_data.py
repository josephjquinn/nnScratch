import pytest
from data import DataProcessor


@pytest.fixture(scope="module")
def data_processor():
    processor = DataProcessor("./data/numeric-mnist.csv")
    processor.split_data()
    processor.get_features_and_labels()
    return processor


def test_data_loading(data_processor):
    assert data_processor.data.shape == (42000, 785)
    assert data_processor.m == 42000
    assert data_processor.n == 785


def test_split_data(data_processor):
    assert data_processor.dev.shape == ((data_processor.m * 0.10), 785)
    assert data_processor.test.shape == ((data_processor.m * 0.10), 785)
    assert data_processor.train.shape == ((data_processor.m * 0.80), 785)


def test_get_features_and_labels(data_processor):
    assert data_processor.y_dev.shape == (4200,)
    assert data_processor.x_dev.shape == (784, 4200)
    assert data_processor.y_test.shape == (4200,)
    assert data_processor.x_test.shape == (784, 4200)
    assert data_processor.y_train.shape == (33600,)
    assert data_processor.x_train.shape == (784, 33600)
