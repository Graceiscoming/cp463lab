"""
activation function และอนุพันธ์ (สไลด์ p.4-6, p.10-12)
Activation functions and their derivatives

ทุกฟังก์ชันรับ numpy array shape ใดก็ได้และคืน array shape เดิม (elementwise)
"""
from __future__ import annotations

from typing import Callable

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """g(z) = 1 / (1 + e^{-z})   ค่าอยู่ใน (0, 1)

    เขียนแบบ numerically stable: ถ้าคำนวณ 1/(1+exp(-z)) ตรงๆ เมื่อ z = -1000
    exp(1000) จะ overflow เป็น inf (มี RuntimeWarning) จึงแยกกรณี z < 0
    ใช้รูป e^{z} / (1 + e^{z}) ซึ่งเท่ากันทางคณิตศาสตร์แต่ไม่ overflow
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def d_sigmoid(z: np.ndarray) -> np.ndarray:
    """g'(z) = g(z)(1 - g(z))   (สไลด์ p.10)"""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    """g(z) = tanh(z)   ค่าอยู่ใน (-1, 1)"""
    return np.tanh(np.asarray(z, dtype=float))


def d_tanh(z: np.ndarray) -> np.ndarray:
    """g'(z) = 1 - tanh²(z)   (สไลด์ p.11)"""
    t = np.tanh(np.asarray(z, dtype=float))
    return 1.0 - t * t


def relu(z: np.ndarray) -> np.ndarray:
    """g(z) = max(0, z)"""
    return np.maximum(0.0, np.asarray(z, dtype=float))


def d_relu(z: np.ndarray) -> np.ndarray:
    """g'(z) = 1 ถ้า z > 0, 0 ถ้า z < 0 (ที่ z = 0 นิยามไม่ได้ ในทางปฏิบัติใช้ 0 — สไลด์ p.12)"""
    return (np.asarray(z, dtype=float) > 0).astype(float)


def linear(z: np.ndarray) -> np.ndarray:
    """g(z) = z  (ไม่มี activation — ใช้กับ regression หรือ logits)"""
    return np.asarray(z, dtype=float)


def d_linear(z: np.ndarray) -> np.ndarray:
    return np.ones_like(np.asarray(z, dtype=float))


def softmax(z: np.ndarray, axis: int = 0) -> np.ndarray:
    """softmax ตามแกน axis: e^{z_i} / Σ e^{z_j}   ผลรวมตามแกนนั้น = 1

    ลบค่า max ออกก่อน exponentiate เพื่อไม่ให้ overflow (ผลลัพธ์ไม่เปลี่ยน)
    axis=0 สำหรับ convention สไลด์ (C, m), axis=1 สำหรับ (m, C)
    """
    z = np.asarray(z, dtype=float)
    shifted = z - z.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def numerical_derivative(
    f: Callable[[np.ndarray], np.ndarray], x: np.ndarray, dx: float = 1e-3, method: str = "forward"
) -> np.ndarray:
    """อนุพันธ์เชิงตัวเลขด้วยการ "nudge" x (สไลด์ p.9: Δf/Δx เมื่อ Δx = 0.001)
    Numerical derivative by nudging x.

    method="forward":  (f(x+dx) - f(x)) / dx            ← แบบในสไลด์ อธิบายง่าย
    method="central":  (f(x+dx) - f(x-dx)) / (2 dx)      ← แม่นกว่า ใช้ตอน gradient check

    >>> numerical_derivative(lambda x: x**2, np.array(2.0))      # สไลด์: 0.004/0.001 = 4
    4.000999...
    """
    x = np.asarray(x, dtype=float)
    if method == "forward":
        return (f(x + dx) - f(x)) / dx
    if method == "central":
        return (f(x + dx) - f(x - dx)) / (2.0 * dx)
    raise ValueError("method ต้องเป็น 'forward' หรือ 'central'")


# ชื่อ → (ฟังก์ชัน, อนุพันธ์) สำหรับให้ NeuralNetwork เลือกด้วย string
ACTIVATIONS: dict[str, tuple[Callable, Callable]] = {
    "sigmoid": (sigmoid, d_sigmoid),
    "tanh": (tanh, d_tanh),
    "relu": (relu, d_relu),
    "linear": (linear, d_linear),
    # softmax ใช้เฉพาะชั้น output คู่กับ cross-entropy ซึ่ง dZ = A - Y จึงไม่ต้องมีอนุพันธ์แยก
    "softmax": (softmax, None),
}
