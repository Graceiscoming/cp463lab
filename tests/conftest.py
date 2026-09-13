"""fixtures ร่วมของทุก test: ข้อมูลตัวอย่างจากสไลด์ในทั้งสอง convention"""
import numpy as np
import pytest


@pytest.fixture
def lung_rows():
    """ตารางสไลด์ p.40 แบบ library convention: X (3, 2), y (3,)"""
    X = np.array([[3.0, 1.0], [5.0, 0.0], [2.0, 1.0]])
    y = np.array([0, 1, 0])
    return X, y


@pytest.fixture
def deck_XY(lung_rows):
    """แบบสไลด์: X (n_x=2, m=3), Y (1, 3)"""
    X, y = lung_rows
    return X.T.copy(), y.reshape(1, -1).astype(float)


@pytest.fixture
def rng():
    return np.random.default_rng(463)
