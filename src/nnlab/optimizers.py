"""
gradient optimizer ตามสไลด์ p.107-114: SGD, Momentum, RMSprop, Adam
Optimizers written exactly as the lecture formulas

ทุกตัวมี method เดียว  step(params, grads)  ที่แก้ params "in-place"
    params = {"W1": ..., "b1": ..., "W2": ..., ...}
    grads  = {"dW1": ..., "db1": ..., "dW2": ..., ...}       (ชื่อ = "d" + ชื่อ parameter)
"""
from __future__ import annotations

import numpy as np


class Optimizer:
    """base class: gradient descent ธรรมดา  θ := θ − α dθ   (สไลด์ p.20-21)"""

    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.t = 0  # จำนวนครั้งที่ step ถูกเรียก (ใช้ใน bias correction)

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for key in params:
            params[key] -= self.lr * grads["d" + key]

    def reset(self) -> None:
        self.t = 0

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in vars(self).items() if not k.startswith("_") and k != "t")
        return f"{type(self).__name__}({args})"


class SGD(Optimizer):
    """gradient descent ธรรมดา (ชื่อ SGD เพราะปกติใช้กับ mini-batch)"""


class Momentum(Optimizer):
    """v := β v + (1 − β) dθ ;  θ := θ − α v      (สไลด์ p.109-110, β = 0.9)

    v คือ "ความเร็ว" ที่สะสมทิศทางของ gradient — แกว่งน้อยลง วิ่งตรงเข้าหา minimum เร็วขึ้น
    หมายเหตุ: torch.optim.SGD(momentum=β) ใช้รูป v := β v + dθ (ไม่มี 1−β) ค่าจึงต่างกันด้วยตัวคูณ (1−β)
    """

    def __init__(self, lr: float = 0.01, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self._v: dict[str, np.ndarray] = {}

    def step(self, params, grads) -> None:
        self.t += 1
        for key in params:
            g = grads["d" + key]
            v = self._v.setdefault(key, np.zeros_like(g))
            v *= self.beta
            v += (1 - self.beta) * g
            params[key] -= self.lr * v

    def reset(self) -> None:
        super().reset()
        self._v.clear()


class RMSProp(Optimizer):
    """s := β s + (1 − β) dθ² ;  θ := θ − α dθ / (√s + ε)      (สไลด์ p.111)

    หารด้วยขนาดของ gradient ที่ผ่านมา → มิติที่ gradient ใหญ่จะก้าวเล็กลง ใช้ learning rate ใหญ่ได้โดยไม่ overshoot
    (torch.optim.RMSprop(alpha=β, eps=ε) ใช้สูตรเดียวกัน)
    """

    def __init__(self, lr: float = 0.001, beta: float = 0.999, eps: float = 1e-8):
        super().__init__(lr)
        self.beta = beta
        self.eps = eps
        self._s: dict[str, np.ndarray] = {}

    def step(self, params, grads) -> None:
        self.t += 1
        for key in params:
            g = grads["d" + key]
            s = self._s.setdefault(key, np.zeros_like(g))
            s *= self.beta
            s += (1 - self.beta) * g * g
            params[key] -= self.lr * g / (np.sqrt(s) + self.eps)

    def reset(self) -> None:
        super().reset()
        self._s.clear()


class Adam(Optimizer):
    """Adam = Momentum + RMSprop + bias correction      (สไลด์ p.112-114)

    v := β₁ v + (1 − β₁) dθ            (momentum)
    s := β₂ s + (1 − β₂) dθ²           (RMSprop)
    v̂ = v / (1 − β₁ᵗ),  ŝ = s / (1 − β₂ᵗ)   (bias correction — แก้ที่ช่วงแรก v, s เริ่มจาก 0 จึงเล็กเกินจริง)
    θ := θ − α v̂ / (√ŝ + ε)
    ค่าแนะนำ: β₁ = 0.9, β₂ = 0.999, ε = 1e-8, α ต้อง tune   (ตรงกับ torch.optim.Adam)
    """

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        super().__init__(lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self._v: dict[str, np.ndarray] = {}
        self._s: dict[str, np.ndarray] = {}

    def step(self, params, grads) -> None:
        self.t += 1
        for key in params:
            g = grads["d" + key]
            v = self._v.setdefault(key, np.zeros_like(g))
            s = self._s.setdefault(key, np.zeros_like(g))
            v *= self.beta1
            v += (1 - self.beta1) * g
            s *= self.beta2
            s += (1 - self.beta2) * g * g
            v_hat = v / (1 - self.beta1**self.t)
            s_hat = s / (1 - self.beta2**self.t)
            params[key] -= self.lr * v_hat / (np.sqrt(s_hat) + self.eps)

    def reset(self) -> None:
        super().reset()
        self._v.clear()
        self._s.clear()


def lr_decay(alpha0: float, epoch: int, decay_rate: float = 0.01) -> float:
    """learning rate decay  α = α₀ / (1 + decay_rate · epoch)   (สไลด์ p.114)"""
    return alpha0 / (1.0 + decay_rate * epoch)


def make_optimizer(name: str, lr: float, **kwargs) -> Optimizer:
    """สร้าง optimizer จากชื่อ: 'sgd' | 'momentum' | 'rmsprop' | 'adam'"""
    table = {"sgd": SGD, "momentum": Momentum, "rmsprop": RMSProp, "adam": Adam}
    if name not in table:
        raise ValueError(f"ไม่รู้จัก optimizer '{name}' — เลือกจาก {list(table)}")
    return table[name](lr=lr, **kwargs)
