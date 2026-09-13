"""
สะพานข้าม convention ของข้อมูลระหว่างสไลด์กับ library
Bridge between the lecture-deck data convention and the library convention

สไลด์ (ตาม Andrew Ng):   X ∈ ℝ^{n_x × m}   sample เป็น "คอลัมน์",  Y ∈ ℝ^{1 × m}
pandas / sklearn / torch: X ∈ ℝ^{m × n_x}   sample เป็น "แถว",     y ∈ ℝ^{m}

ทั้งสองแบบเก็บข้อมูลเดียวกัน ต่างกันแค่ transpose แต่ถ้าสลับกันโดยไม่รู้ตัว
สูตรทุกสูตรจะพัง (หรือแย่กว่านั้นคือรันผ่านแต่ผลผิด เมื่อ m == n_x)
"""
from __future__ import annotations

import numpy as np


def to_deck(X_rows: np.ndarray, y: np.ndarray | None = None):
    """แปลงจาก library convention (m, n_x) → สไลด์ (n_x, m)
    Convert row-major (m, n_x) data to the deck's column-major (n_x, m) layout.

    y (m,) จะกลายเป็น Y (1, m) ด้วย

    >>> X_rows = np.array([[3, 1], [5, 0], [2, 1]])      # 3 samples, 2 features
    >>> X, Y = to_deck(X_rows, np.array([0, 1, 0]))
    >>> X.shape, Y.shape
    ((2, 3), (1, 3))
    """
    X_rows = np.asarray(X_rows, dtype=float)
    if X_rows.ndim == 1:  # sample เดียว → ทำให้เป็น (1, n_x) ก่อน
        X_rows = X_rows.reshape(1, -1)
    X_cols = X_rows.T
    if y is None:
        return X_cols
    Y = np.asarray(y, dtype=float).reshape(1, -1)
    return X_cols, Y


def to_lib(X_cols: np.ndarray, Y: np.ndarray | None = None):
    """แปลงกลับจากสไลด์ (n_x, m) → library convention (m, n_x)
    Inverse of `to_deck`.
    """
    X_rows = np.asarray(X_cols, dtype=float).T
    if Y is None:
        return X_rows
    y = np.asarray(Y).reshape(-1)
    return X_rows, y


def assert_deck(X: np.ndarray, n_x: int | None = None, m: int | None = None) -> None:
    """ตรวจว่า X อยู่ใน convention ของสไลด์ (n_x, m) จริง ไม่งั้น raise พร้อมข้อความช่วย
    Fail loudly when X is not laid out as (n_x, m).
    """
    if X.ndim != 2:
        raise ValueError(f"X ต้องเป็น 2 มิติ (n_x, m) แต่ได้ ndim={X.ndim} shape={X.shape}")
    if n_x is not None and X.shape[0] != n_x:
        raise ValueError(
            f"คาดว่า X.shape[0] (n_x) = {n_x} แต่ได้ {X.shape[0]} — "
            f"X.shape={X.shape} อาจยังเป็น (m, n_x) อยู่ ลอง to_deck(X) ก่อน"
        )
    if m is not None and X.shape[1] != m:
        raise ValueError(f"คาดว่า X.shape[1] (m) = {m} แต่ได้ {X.shape[1]} — X.shape={X.shape}")


def describe(X: np.ndarray, convention: str = "deck") -> str:
    """สร้างข้อความอธิบาย shape ให้พิมพ์ใน notebook / human-readable shape description"""
    X = np.asarray(X)
    if X.ndim != 2:
        return f"shape {X.shape} (ndim={X.ndim})"
    if convention == "deck":
        return f"shape {X.shape} → n_x={X.shape[0]} features, m={X.shape[1]} samples (sample เป็นคอลัมน์)"
    return f"shape {X.shape} → m={X.shape[0]} samples, n_x={X.shape[1]} features (sample เป็นแถว)"
