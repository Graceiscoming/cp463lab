"""
การเตรียมข้อมูล: one-hot encoding (สไลด์ p.71-73) และ input normalization (p.98-100)
Data preparation: one-hot encoding and standardization
"""
from __future__ import annotations

import numpy as np


def one_hot(labels: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """แปลง label จำนวนเต็ม (m,) → matrix (m, C) ที่แต่ละแถวมีเลข 1 ตัวเดียว

    >>> one_hot(np.array([0, 2, 1]), 3)
    array([[1, 0, 0],
           [0, 0, 1],
           [0, 1, 0]])
    """
    labels = np.asarray(labels).astype(int).reshape(-1)
    if n_classes is None:
        n_classes = int(labels.max()) + 1
    out = np.zeros((labels.shape[0], n_classes), dtype=int)
    out[np.arange(labels.shape[0]), labels] = 1
    return out


def one_hot_dataframe(df, columns: list[str] | None = None, drop_first: bool = False):
    """one-hot คอลัมน์ categorical ของ DataFrame (ห่อ pd.get_dummies ให้ได้ 0/1 เป็นตัวเลข)

    ตัวอย่างสไลด์ p.73: Subscription Type {Basic, Standard, Premium} → 3 คอลัมน์ 0/1
    columns=None → เลือกทุกคอลัมน์ที่เป็น object/category อัตโนมัติ
    """
    import pandas as pd

    if columns is None:
        columns = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == "category"]
    return pd.get_dummies(df, columns=list(columns), dtype=int, drop_first=drop_first)


class StandardNormalizer:
    """normalize input ด้วย μ และ σ ที่คำนวณจาก training set เท่านั้น (สไลด์ p.98-99)
    Standardize features using train-set statistics only.

    X := (X - μ) / σ   ทำให้ทุก feature มี mean 0, variance 1
    สำคัญ: fit บน train แล้ว transform ทั้ง train และ test ด้วย μ, σ เดียวกัน
    (ห้าม fit ใหม่บน test — ไม่งั้นข้อมูล test "รั่ว" เข้ามาในการเทรน)

    X มี shape (m, n_x) ตาม library convention; axis=0 คือเฉลี่ยข้าม sample
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardNormalizer":
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)          # ddof=0 ตรงกับ σ² = (1/m) Σ x² ในสไลด์
        self.std_ = np.where(self.std_ < self.eps, 1.0, self.std_)  # feature ค่าคงที่ → ไม่หารศูนย์
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("ต้องเรียก fit(X_train) ก่อน transform")
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return np.asarray(X_scaled, dtype=float) * self.std_ + self.mean_
