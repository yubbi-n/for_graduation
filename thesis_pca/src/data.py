"""MNIST 로드, 표준 분할(60,000 / 10,000), [0, 1] 정규화."""
import numpy as np
from sklearn.datasets import fetch_openml

from . import config


def load_mnist():
    """(X_train, X_test, y_train, y_test)를 반환한다.

    픽셀값은 255.0으로 나누어 [0, 1]로 정규화하며 StandardScaler는 사용하지 않는다.
    """
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto", return_X_y=True)
    X = X.astype(np.float64) / 255.0
    y = y.astype(int)

    n_train = config.N_TRAIN
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    assert X_test.shape[0] == config.N_TEST
    return X_train, X_test, y_train, y_test
