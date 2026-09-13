"""
perceptron / logistic regression ด้วย numpy ตามสไลด์ p.13-31 (vectorized) และ p.29 (naive loop)
A single sigmoid perceptron trained with gradient descent

convention ภายในไฟล์นี้ = สไลด์:
    X  (n_x, m)   sample เป็นคอลัมน์
    Y  (1, m)
    w  (n_x, 1)   ← ตรึงให้เป็นคอลัมน์เสมอ (สไลด์บางหน้าเขียน 1×n_x แต่สูตร wᵀ·x ต้องการ n_x×1)
    b  scalar
class Perceptron รับ X แบบ (m, n_x) ที่ขอบเขต (library convention) แล้วเรียก to_deck ภายใน
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .activations import sigmoid
from .conventions import to_deck
from .losses import binary_cross_entropy, l2_penalty
from .metrics import binary_report


# =============================================================================
# ฟังก์ชันระดับล่างที่ notebook เรียกดูได้ทีละขั้น
# =============================================================================
def forward(w: np.ndarray, b: float, X: np.ndarray):
    """Z = wᵀ·X + b ,  A = σ(Z)      (สไลด์ p.28, p.30)

    w (n_x, 1), X (n_x, m) → Z, A มี shape (1, m)
    """
    Z = w.T @ X + b                 # (1, n_x) @ (n_x, m) → (1, m)  b broadcast เป็น (1, m)
    A = sigmoid(Z)
    return A, Z


def backward(X: np.ndarray, A: np.ndarray, Y: np.ndarray):
    """dz = A − Y ,  dw = (1/m) X·dzᵀ ,  db = (1/m) Σ dz      (สไลด์ p.26, p.30-31)"""
    m = X.shape[1]
    dz = A - Y                      # (1, m)
    dw = (X @ dz.T) / m             # (n_x, m) @ (m, 1) → (n_x, 1)
    db = float(np.sum(dz) / m)
    return dw, db, dz


def naive_epoch(w: np.ndarray, b: float, X: np.ndarray, Y: np.ndarray, lr: float):
    """หนึ่ง epoch แบบ "loop and loop and loop" ตามสไลด์ p.29 — ช้าแต่เห็นทุกขั้น

    วน i = 1..m คำนวณทีละ sample แล้ววน j = 1..n_x สะสม dw_j
    คืน (w_new, b_new, cost) — ผลต้องเท่ากับ vectorized ทุกหลัก
    """
    n_x, m = X.shape
    J = 0.0
    dw = np.zeros((n_x, 1))
    db = 0.0
    for i in range(m):
        z_i = b
        for j in range(n_x):
            z_i += w[j, 0] * X[j, i]
        a_i = 1.0 / (1.0 + np.exp(-z_i))
        a_c = min(max(a_i, 1e-12), 1 - 1e-12)
        J += -(Y[0, i] * np.log(a_c) + (1 - Y[0, i]) * np.log(1 - a_c))
        dz_i = a_i - Y[0, i]
        for j in range(n_x):
            dw[j, 0] += X[j, i] * dz_i
        db += dz_i
    J /= m
    dw /= m
    db /= m
    return w - lr * dw, b - lr * db, J


def gradient_check(w: np.ndarray, b: float, X: np.ndarray, Y: np.ndarray, eps: float = 1e-7) -> float:
    """เทียบ gradient จาก backward กับอนุพันธ์เชิงตัวเลข (nudge ทีละ parameter) คืนค่าความต่างสัมพัทธ์

    ค่าที่ดีควร < 1e-7 ; ถ้า > 1e-3 แสดงว่า backward ผิด
    """
    A, _ = forward(w, b, X)
    dw, db, _ = backward(X, A, Y)
    analytic = np.concatenate([dw.ravel(), [db]])
    theta = np.concatenate([w.ravel(), [b]])
    numeric = np.zeros_like(theta)
    for k in range(theta.size):
        plus, minus = theta.copy(), theta.copy()
        plus[k] += eps
        minus[k] -= eps
        Jp = binary_cross_entropy(forward(plus[:-1].reshape(-1, 1), plus[-1], X)[0], Y)
        Jm = binary_cross_entropy(forward(minus[:-1].reshape(-1, 1), minus[-1], X)[0], Y)
        numeric[k] = (Jp - Jm) / (2 * eps)
    return float(np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12))


# =============================================================================
# โมเดลแบบ production: interface เดียวกับ sklearn / torch wrapper
# =============================================================================
class Perceptron:
    """perceptron หนึ่งตัว (sigmoid) = logistic regression เทรนด้วย batch gradient descent

    พารามิเตอร์
        lr      learning rate α
        epochs  จำนวนรอบที่วนทั้ง training set
        l2      λ ของ L2 regularization (0 = ไม่ใช้)
        init    "zeros" (สไลด์) หรือ "random"
    หลัง fit:
        w_ (n_x, 1), b_ float, history_ {"cost": [...]}
    """

    def __init__(self, lr: float = 0.1, epochs: int = 1000, l2: float = 0.0, init: str = "zeros", seed: int = 463, verbose: bool = False):
        self.lr, self.epochs, self.l2, self.init, self.seed, self.verbose = lr, epochs, l2, init, seed, verbose
        self.w_: np.ndarray | None = None
        self.b_: float = 0.0
        self.history_: dict[str, list[float]] = {"cost": []}

    # ----- การเทรน ------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        X_col, Y = to_deck(X, y)                     # (n_x, m), (1, m)
        n_x, m = X_col.shape
        rng = np.random.default_rng(self.seed)
        self.w_ = np.zeros((n_x, 1)) if self.init == "zeros" else rng.normal(0, 0.01, (n_x, 1))
        self.b_ = 0.0
        self.history_ = {"cost": []}
        for epoch in range(self.epochs):
            A, _ = forward(self.w_, self.b_, X_col)
            cost = binary_cross_entropy(A, Y) + l2_penalty([self.w_], self.l2, m)
            dw, db, _ = backward(X_col, A, Y)
            if self.l2:
                dw = dw + (self.l2 / m) * self.w_    # weight decay term (สไลด์ p.86)
            self.w_ -= self.lr * dw
            self.b_ -= self.lr * db
            self.history_["cost"].append(cost)
            if self.verbose and (epoch % max(1, self.epochs // 10) == 0):
                print(f"epoch {epoch:5d}  cost {cost:.4f}")
        return self

    # ----- การทำนาย -----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w_ is None:
            raise RuntimeError("ต้อง fit ก่อน predict")
        A, _ = forward(self.w_, self.b_, to_deck(X))
        return A.reshape(-1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        return binary_report(y, self.predict(X))

    # ----- บันทึก/โหลด --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w=self.w_, b=self.b_, lr=self.lr, epochs=self.epochs, l2=self.l2, cost=np.array(self.history_["cost"]))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Perceptron":
        data = np.load(Path(path).with_suffix(".npz"))
        model = cls(lr=float(data["lr"]), epochs=int(data["epochs"]), l2=float(data["l2"]))
        model.w_ = data["w"]
        model.b_ = float(data["b"])
        model.history_ = {"cost": data["cost"].tolist()}
        return model

    def __repr__(self) -> str:
        return f"Perceptron(lr={self.lr}, epochs={self.epochs}, l2={self.l2}, fitted={self.w_ is not None})"
