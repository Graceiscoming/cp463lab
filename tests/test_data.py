import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nnlab.data import (
    get_dataset, kfold_indices, load_blobs, load_breast_cancer_scaled, load_churn_synthetic, load_churn_toy,
    load_digits_images, load_lung_cancer_toy, load_moons, stratified_split,
)

LAB = Path(__file__).resolve().parents[1]


def test_make_data_is_deterministic():
    result = subprocess.run([sys.executable, str(LAB / "scripts" / "make_data.py"), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_lung_toy_matches_slide_p40():
    X, y = load_lung_cancer_toy()
    assert X.tolist() == [[3, 1], [5, 0], [2, 1]] and y.tolist() == [0, 1, 0]


def test_churn_toy_matches_slide_p72():
    df = load_churn_toy()
    assert len(df) == 5
    assert df["subscription_type"].tolist() == ["Basic", "Standard", "Premium", "Standard", "Basic"]
    assert df["churn"].tolist() == ["Yes", "No", "Yes", "No", "Yes"]


def test_churn_synthetic_shape_and_rate():
    df = load_churn_synthetic(as_frame=True)
    assert len(df) == 1000
    assert 0.25 < df["churn"].mean() < 0.35
    X, y, names = load_churn_synthetic()
    assert X.shape == (1000, 6) and len(names) == 6          # 3 numeric + 3 one-hot (customer_id ถูกตัด? ไม่ — ดูด้านล่าง)


def test_stratified_split_keeps_class_ratio():
    X, y = load_moons(n=500)
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, 0.2, seed=1)
    assert len(y_te) == 100 and len(y_tr) == 400
    assert y_te.mean() == pytest.approx(y.mean(), abs=0.01)
    assert set(map(tuple, X_te)).isdisjoint(set(map(tuple, X_tr)))


def test_kfold_covers_every_sample_once():
    y = np.array([0] * 30 + [1] * 20)
    folds = kfold_indices(y, k=5)
    all_val = np.concatenate([v for _, v in folds])
    assert sorted(all_val.tolist()) == list(range(50))
    for tr, va in folds:
        assert len(np.intersect1d(tr, va)) == 0
        assert abs(y[va].mean() - 0.4) < 0.15


def test_sklearn_loaders():
    X_tr, X_te, y_tr, y_te = load_breast_cancer_scaled()
    assert X_tr.shape[1] == 30 and len(y_tr) + len(y_te) == 569
    assert np.allclose(X_tr.mean(axis=0), 0, atol=1e-9)
    images, y = load_digits_images(classes=[3, 8])
    assert images.shape[1:] == (8, 8) and set(np.unique(y)) == {0, 1}
    assert load_blobs()[0].shape[1] == 2


@pytest.mark.parametrize("name", ["lung", "churn", "breast_cancer", "moons", "blobs", "digits"])
def test_get_dataset(name):
    X_tr, X_te, y_tr, y_te, n_classes = get_dataset(name)
    assert X_tr.shape[1] == X_te.shape[1] and len(X_tr) == len(y_tr)
    assert n_classes == (10 if name == "digits" else 2)
