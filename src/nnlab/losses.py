"""
loss / cost function (สไลด์ p.17-18) และ regularization term (p.84-86)
Loss and cost functions plus the L2 penalty

คำศัพท์ตามสไลด์:
    loss  ℒ(ŷ, y)  = ความผิดพลาดของ "หนึ่ง" training sample
    cost  𝒥(w, b) = ค่าเฉลี่ยของ loss ทั้ง m samples
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def bce_loss(a: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """binary cross-entropy ต่อ sample: ℒ = -(y log a + (1-y) log(1-a))   (p.17)

    clip a ให้อยู่ใน [eps, 1-eps] ก่อน เพราะ log(0) = -inf จะทำให้ cost เป็น nan/inf
    """
    a = np.clip(np.asarray(a, dtype=float), eps, 1.0 - eps)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(a) + (1.0 - y) * np.log(1.0 - a))


def binary_cross_entropy(A: np.ndarray, Y: np.ndarray, eps: float = 1e-12) -> float:
    """cost 𝒥 = (1/m) Σ ℒ(a⁽ⁱ⁾, y⁽ⁱ⁾)   (p.18)   ใช้ได้ทั้ง shape (1, m) และ (m,)"""
    return float(np.mean(bce_loss(A, Y, eps)))


def cross_entropy(P: np.ndarray, Y_onehot: np.ndarray, eps: float = 1e-12, axis: int = 0) -> float:
    """categorical cross-entropy สำหรับ softmax output: -(1/m) Σ_i Σ_c y_c log p_c

    P และ Y_onehot มี shape (C, m) เมื่อ axis=0 (convention สไลด์) หรือ (m, C) เมื่อ axis=1
    """
    P = np.clip(np.asarray(P, dtype=float), eps, 1.0)
    per_sample = -(np.asarray(Y_onehot, dtype=float) * np.log(P)).sum(axis=axis)
    return float(per_sample.mean())


def l2_penalty(weights: Mapping[str, np.ndarray] | Iterable[np.ndarray], lam: float, m: int) -> float:
    """regularization term  (λ / 2m) Σ_l ‖W[l]‖²_F   (p.85-86)

    weights: dict {"W1": ..., "b1": ...} (จะเลือกเฉพาะ key ที่ขึ้นต้นด้วย W) หรือ list ของ matrix
    ไม่ regularize bias — เป็นธรรมเนียมทั่วไป
    """
    if lam == 0:
        return 0.0
    if isinstance(weights, Mapping):
        mats = [v for k, v in weights.items() if k.startswith("W")]
    else:
        mats = list(weights)
    return float(lam / (2.0 * m) * sum(np.sum(W * W) for W in mats))
