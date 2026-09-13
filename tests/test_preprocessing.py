import numpy as np
import pandas as pd

from nnlab.preprocessing import StandardNormalizer, one_hot, one_hot_dataframe


def test_one_hot_rows():
    out = one_hot(np.array([0, 2, 1]), 3)
    assert out.tolist() == [[1, 0, 0], [0, 0, 1], [0, 1, 0]]
    assert one_hot(np.array([1, 0])).shape == (2, 2)


def test_one_hot_dataframe_matches_slide_p73():
    df = pd.DataFrame({"subscription_type": ["Basic", "Standard", "Premium", "Standard", "Basic"]})
    out = one_hot_dataframe(df)
    assert list(out.columns) == ["subscription_type_Basic", "subscription_type_Premium", "subscription_type_Standard"]
    assert out["subscription_type_Basic"].tolist() == [1, 0, 0, 0, 1]
    assert out["subscription_type_Standard"].tolist() == [0, 1, 0, 1, 0]
    assert out["subscription_type_Premium"].tolist() == [0, 0, 1, 0, 0]


def test_normalizer_uses_train_statistics_only(rng):
    X_train = rng.normal(5, 2, (100, 3))
    X_test = rng.normal(5, 2, (20, 3))
    norm = StandardNormalizer().fit(X_train)
    Z = norm.transform(X_train)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-12) and np.allclose(Z.std(axis=0), 1, atol=1e-12)
    Z_test = norm.transform(X_test)
    assert not np.allclose(Z_test.mean(axis=0), 0, atol=1e-3)     # test ไม่ได้ถูก fit ใหม่
    assert np.allclose(norm.inverse_transform(Z), X_train)


def test_normalizer_constant_feature_does_not_divide_by_zero():
    X = np.array([[1.0, 5.0], [1.0, 7.0]])
    Z = StandardNormalizer().fit_transform(X)
    assert np.isfinite(Z).all() and np.allclose(Z[:, 0], 0)
