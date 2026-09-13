"""
โมเดลสำเร็จรูปของ scikit-learn ห่อให้มี interface เดียวกับ nnlab.Perceptron / NeuralNetwork
scikit-learn baselines wrapped in the shared Classifier interface

ทำไมต้องห่อ: sklearn ใช้ชื่อ method เหมือนกันอยู่แล้ว (fit/predict/predict_proba) แต่
    - predict_proba คืน (m, 2) ไม่ใช่ (m,)
    - ไม่มี evaluate / save / history_
การห่อทำให้ scripts/train.py สลับโมเดลได้โดยไม่ต้องแก้โค้ดส่วนอื่น
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .metrics import binary_report, classification_summary


class _SklearnWrapper:
    estimator = None  # กำหนดใน subclass

    def __init__(self):
        self.history_: dict[str, list[float]] = {"cost": []}
        self.n_classes_: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes_ = int(len(np.unique(y)))
        self.estimator.fit(X, y)
        curve = getattr(self.estimator, "loss_curve_", None)     # MLPClassifier มี, LogisticRegression ไม่มี
        self.history_ = {"cost": list(curve) if curve is not None else []}
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        P = self.estimator.predict_proba(X)
        return P[:, 1] if self.n_classes_ == 2 else P

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        P = self.predict_proba(X)
        return (P >= threshold).astype(int) if P.ndim == 1 else P.argmax(axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        y_pred = self.predict(X)
        return binary_report(y, y_pred) if self.n_classes_ == 2 else classification_summary(y, y_pred)

    def save(self, path: str | Path) -> Path:
        import joblib

        path = Path(path).with_suffix(".joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: str | Path):
        import joblib

        return joblib.load(Path(path).with_suffix(".joblib"))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.estimator!r})"


class SklearnLogReg(_SklearnWrapper):
    """LogisticRegression ของ sklearn = perceptron sigmoid ตัวเดียว แต่ optimize ด้วย lbfgs และมี L2 (C) โดย default

    C = 1/λ  → C ใหญ่ = regularize น้อย ; ใช้ C=1e6 ถ้าต้องการเทียบกับ Perceptron ที่ไม่มี L2
    coef_ (1, n_x) และ intercept_ เทียบกับ w_ (n_x, 1) และ b_ ของ nnlab.Perceptron ได้โดย transpose
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, seed: int = 463):
        super().__init__()
        from sklearn.linear_model import LogisticRegression

        self.estimator = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)

    @property
    def w_(self) -> np.ndarray:
        return self.estimator.coef_.T

    @property
    def b_(self) -> float:
        return float(self.estimator.intercept_[0])


class SklearnMLP(_SklearnWrapper):
    """MLPClassifier = neural network หลายชั้นของ sklearn (adam by default, มี early stopping ในตัว)"""

    def __init__(self, hidden: tuple[int, ...] = (16,), lr: float = 0.001, epochs: int = 300, alpha: float = 0.0001, batch_size: int | str = "auto", seed: int = 463):
        super().__init__()
        from sklearn.neural_network import MLPClassifier

        self.estimator = MLPClassifier(
            hidden_layer_sizes=tuple(hidden), learning_rate_init=lr, max_iter=epochs, alpha=alpha,
            batch_size=batch_size, random_state=seed,
        )
