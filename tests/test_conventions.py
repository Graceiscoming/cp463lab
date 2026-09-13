import numpy as np
import pytest

from nnlab.conventions import assert_deck, describe, to_deck, to_lib


def test_to_deck_shapes(lung_rows):
    X, y = lung_rows
    X_col, Y = to_deck(X, y)
    assert X_col.shape == (2, 3)
    assert Y.shape == (1, 3)
    assert X_col[0, 1] == 5.0       # feature 0 ของ sample 1 = smoking ของคนที่ 2


def test_round_trip(lung_rows):
    X, y = lung_rows
    X_back, y_back = to_lib(*to_deck(X, y))
    assert np.array_equal(X_back, X)
    assert np.array_equal(y_back, y)


def test_single_sample_becomes_column():
    assert to_deck(np.array([1.0, 2.0, 3.0])).shape == (3, 1)


def test_assert_deck_catches_wrong_layout(lung_rows):
    X, _ = lung_rows
    with pytest.raises(ValueError):
        assert_deck(X, n_x=2, m=3)          # ยังเป็น (m, n_x)
    assert_deck(X.T, n_x=2, m=3)            # ผ่าน


def test_describe_mentions_dimensions(lung_rows):
    X, _ = lung_rows
    assert "m=3" in describe(X, "lib") and "n_x=2" in describe(X, "lib")
