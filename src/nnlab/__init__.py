"""
nnlab — package ประกอบ lab tutorial ของ CP463 (perceptron, neural network, CNN)
nnlab — companion package for the CP463 lab tutorials

โมดูลทุกตัวเขียนให้ notebook import แล้วเทียบผลกับโค้ดที่นิสิตเขียนเองใน cell ได้
ทุกโมเดล (numpy / scikit-learn / PyTorch) ใช้ interface เดียวกันตาม `Classifier` ด้านล่าง

Convention ของข้อมูลที่ขอบเขต public API:
    X มี shape (m, n_x)  — sample เป็นแถว (เหมือน pandas / scikit-learn / PyTorch)
    y มี shape (m,)
โมเดล numpy ภายในจะแปลงเป็น convention ของสไลด์ (n_x, m) ด้วย `nnlab.conventions.to_deck`
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__version__ = "0.1.0"


@runtime_checkable
class Classifier(Protocol):
    """Interface ร่วมของทุกโมเดลใน nnlab / shared interface of every nnlab model

    fit(X, y)            : เทรนด้วย X (m, n_x), y (m,) แล้วคืน self
    predict_proba(X)     : ความน่าจะเป็นของ class 1 → shape (m,)  (multiclass → (m, C))
    predict(X, threshold): label 0/1 → shape (m,)  (multiclass → argmax)
    evaluate(X, y)       : dict ของ metrics (accuracy, precision, recall, f1, ...)
    save(path) / load(path)
    history_             : cost ต่อ epoch สำหรับ plot
    """

    history_: dict[str, list[float]]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Classifier": ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray: ...
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]: ...
    def save(self, path) -> None: ...
