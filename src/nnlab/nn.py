"""
neural network L ชั้นด้วย numpy ตามสไลด์ p.74-80 (+ L2 p.86, dropout p.91-95, mini-batch p.101-106)
An L-layer fully-connected network with forward/backward exactly as the lecture

notation ของสไลด์ (ใช้ตรงๆ ในโค้ด):
    L        จำนวนชั้น (ไม่นับ input)
    n[l]     จำนวน unit ของชั้น l          → layer_sizes = [n[0]=n_x, n[1], ..., n[L]]
    W[l]     shape (n[l], n[l-1])           → params["W1"], params["W2"], ...
    b[l]     shape (n[l], 1)                → params["b1"], ...
    Z[l] = W[l]·A[l-1] + b[l] ,  A[l] = g[l](Z[l]) ,  A[0] = X (n_x, m)
backward (p.80):
    dZ[l]   = dA[l] * g'[l](Z[l])
    dW[l]   = (1/m) dZ[l]·A[l-1]ᵀ
    db[l]   = (1/m) Σ dZ[l]   (np.sum(..., axis=1, keepdims=True))
    dA[l-1] = W[l]ᵀ·dZ[l]
ชั้น output: sigmoid + BCE (binary) หรือ softmax + cross-entropy (multiclass) → dZ[L] = A[L] − Y ทั้งคู่
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .activations import ACTIVATIONS
from .conventions import to_deck
from .losses import binary_cross_entropy, cross_entropy, l2_penalty
from .metrics import binary_report, classification_summary
from .optimizers import Optimizer, make_optimizer
from .preprocessing import one_hot


# =============================================================================
# ฟังก์ชันระดับล่าง (notebook lab07 เรียกทีละขั้น)
# =============================================================================
def init_params(layer_sizes: list[int], seed: int = 463, scale: str | float = "he") -> dict[str, np.ndarray]:
    """สร้าง W[l] (n[l], n[l-1]) แบบสุ่ม และ b[l] (n[l], 1) เป็นศูนย์ (สไลด์ p.77-78)

    scale="he"  : W ~ N(0, 2/n[l-1])  เหมาะกับ ReLU
    scale=0.01  : W ~ N(0, 1) × 0.01  แบบง่ายที่สุด
    W ต้องไม่เป็นศูนย์ทั้งหมด ไม่งั้นทุก unit ในชั้นเดียวกันจะเรียนเหมือนกันหมด (symmetry)
    """
    rng = np.random.default_rng(seed)
    params: dict[str, np.ndarray] = {}
    for l in range(1, len(layer_sizes)):
        n_l, n_prev = layer_sizes[l], layer_sizes[l - 1]
        factor = np.sqrt(2.0 / n_prev) if scale == "he" else float(scale)
        params[f"W{l}"] = rng.standard_normal((n_l, n_prev)) * factor
        params[f"b{l}"] = np.zeros((n_l, 1))
    return params


def n_layers(params: dict[str, np.ndarray]) -> int:
    return len([k for k in params if k.startswith("W")])


def forward(
    X: np.ndarray,
    params: dict[str, np.ndarray],
    activations: list[str],
    keep_prob: float = 1.0,
    training: bool = False,
    rng: np.random.Generator | None = None,
):
    """forward propagation ทุกชั้น คืน (A[L], caches)

    caches[l-1] = dict(A_prev, Z, A, D) ของชั้น l — เก็บไว้ใช้ตอน backward (สไลด์ p.79 "cache")
    dropout (p.91-95): ใช้เฉพาะ training=True กับ hidden layer; D คือ mask 0/1 และหาร keep_prob (inverted dropout)
    เพื่อให้ค่าคาดหวังของ A เท่าเดิม ตอน predict จึงไม่ต้องทำอะไรเพิ่ม
    """
    L = n_layers(params)
    A = X
    caches = []
    rng = rng or np.random.default_rng()
    for l in range(1, L + 1):
        g, _ = ACTIVATIONS[activations[l - 1]]
        A_prev = A
        Z = params[f"W{l}"] @ A_prev + params[f"b{l}"]          # (n[l], n[l-1]) @ (n[l-1], m) + (n[l], 1)
        A = g(Z, axis=0) if activations[l - 1] == "softmax" else g(Z)
        D = None
        if training and keep_prob < 1.0 and l < L:                # dropout เฉพาะ hidden layer ตอนเทรน
            D = (rng.random(A.shape) < keep_prob).astype(float)
            A = A * D / keep_prob
        caches.append({"A_prev": A_prev, "Z": Z, "A": A, "D": D})
    return A, caches


def compute_cost(A_L: np.ndarray, Y: np.ndarray, params: dict[str, np.ndarray] | None = None, l2: float = 0.0) -> float:
    """cost = BCE (ถ้า output 1 unit) หรือ cross-entropy (softmax) + L2 term"""
    m = Y.shape[1]
    base = binary_cross_entropy(A_L, Y) if A_L.shape[0] == 1 else cross_entropy(A_L, Y, axis=0)
    return base + (l2_penalty(params, l2, m) if params is not None and l2 else 0.0)


def backward(
    Y: np.ndarray,
    caches: list[dict],
    params: dict[str, np.ndarray],
    activations: list[str],
    l2: float = 0.0,
    keep_prob: float = 1.0,
) -> dict[str, np.ndarray]:
    """backward propagation ทุกชั้น ตามสูตรสไลด์ p.80 คืน grads {"dW1", "db1", ...}"""
    L = len(caches)
    m = Y.shape[1]
    grads: dict[str, np.ndarray] = {}
    dZ = caches[-1]["A"] - Y                                    # ชั้น output: sigmoid+BCE หรือ softmax+CE → A − Y
    for l in range(L, 0, -1):
        cache = caches[l - 1]
        grads[f"dW{l}"] = (dZ @ cache["A_prev"].T) / m + (l2 / m) * params[f"W{l}"]
        grads[f"db{l}"] = np.sum(dZ, axis=1, keepdims=True) / m   # keepdims → (n[l], 1) ไม่ใช่ (n[l],)
        if l > 1:
            dA_prev = params[f"W{l}"].T @ dZ                     # (n[l-1], n[l]) @ (n[l], m)
            prev = caches[l - 2]
            if prev["D"] is not None:                             # ส่ง gradient ผ่านเฉพาะ unit ที่ไม่ถูก drop
                dA_prev = dA_prev * prev["D"] / keep_prob
            _, dg = ACTIVATIONS[activations[l - 2]]
            dZ = dA_prev * dg(prev["Z"])
    return grads


def gradient_check(
    X: np.ndarray, Y: np.ndarray, params: dict[str, np.ndarray], activations: list[str], l2: float = 0.0, eps: float = 1e-7
) -> float:
    """เทียบ grads จาก backward กับอนุพันธ์เชิงตัวเลขของ cost ทีละ parameter คืนความต่างสัมพัทธ์ (ควร < 1e-6)"""
    A_L, caches = forward(X, params, activations)
    grads = backward(Y, caches, params, activations, l2)
    keys = sorted(params)
    analytic = np.concatenate([grads["d" + k].ravel() for k in keys])
    numeric = np.zeros_like(analytic)
    idx = 0
    for k in keys:
        flat = params[k].ravel()
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + eps
            Jp = compute_cost(forward(X, params, activations)[0], Y, params, l2)
            flat[i] = old - eps
            Jm = compute_cost(forward(X, params, activations)[0], Y, params, l2)
            flat[i] = old
            numeric[idx] = (Jp - Jm) / (2 * eps)
            idx += 1
    return float(np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric) + 1e-12))


def iterate_minibatches(X: np.ndarray, Y: np.ndarray, batch_size: int | None, rng: np.random.Generator):
    """แบ่ง X (n_x, m), Y (C, m) เป็น mini-batch X{t}, Y{t} ตามสไลด์ p.102 หลังสับลำดับ sample"""
    m = X.shape[1]
    if batch_size is None or batch_size >= m:
        yield X, Y
        return
    perm = rng.permutation(m)
    for start in range(0, m, batch_size):
        idx = perm[start : start + batch_size]
        yield X[:, idx], Y[:, idx]


# =============================================================================
# โมเดลแบบ production
# =============================================================================
class NeuralNetwork:
    """fully-connected network L ชั้น เทรนด้วย (mini-batch) gradient descent

    layer_sizes  [n_x, n[1], ..., n[L]]  เช่น [2, 8, 1] = 2-layer network (1 hidden layer)
    activations  ชื่อ activation ต่อชั้น (ยาว L) default: relu ทุก hidden + sigmoid/softmax ที่ output
    lr, epochs, batch_size (None = batch GD), l2 (λ), keep_prob (dropout), optimizer ('sgd'|'momentum'|'rmsprop'|'adam' หรือ object)
    early_stopping_patience  หยุดเมื่อ val cost ไม่ดีขึ้นติดกัน N epoch (ต้องส่ง X_val, y_val ให้ fit)
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[str] | None = None,
        lr: float = 0.05,
        epochs: int = 1000,
        batch_size: int | None = None,
        l2: float = 0.0,
        keep_prob: float = 1.0,
        optimizer: str | Optimizer = "sgd",
        early_stopping_patience: int | None = None,
        seed: int = 463,
        init_scale: str | float = "he",
        verbose: bool = False,
    ):
        self.layer_sizes = list(layer_sizes)
        L = len(self.layer_sizes) - 1
        if activations is None:
            out = "sigmoid" if self.layer_sizes[-1] == 1 else "softmax"
            activations = ["relu"] * (L - 1) + [out]
        if len(activations) != L:
            raise ValueError(f"activations ต้องยาว {L} (จำนวนชั้นไม่นับ input) แต่ได้ {len(activations)}")
        self.activations = list(activations)
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.l2, self.keep_prob = l2, keep_prob
        self.optimizer = make_optimizer(optimizer, lr) if isinstance(optimizer, str) else optimizer
        self.early_stopping_patience = early_stopping_patience
        self.seed, self.init_scale, self.verbose = seed, init_scale, verbose
        self.params_: dict[str, np.ndarray] = {}
        self.history_: dict[str, list[float]] = {"cost": [], "val_cost": []}

    @property
    def n_classes(self) -> int:
        return 2 if self.layer_sizes[-1] == 1 else self.layer_sizes[-1]

    def _targets(self, y: np.ndarray) -> np.ndarray:
        """y (m,) → Y (1, m) สำหรับ binary หรือ (C, m) one-hot สำหรับ multiclass"""
        y = np.asarray(y).reshape(-1)
        if self.layer_sizes[-1] == 1:
            return y.astype(float).reshape(1, -1)
        return one_hot(y, self.layer_sizes[-1]).T.astype(float)

    # ----- การเทรน ------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "NeuralNetwork":
        X_col = to_deck(X)
        Y = self._targets(y)
        if X_col.shape[0] != self.layer_sizes[0]:
            raise ValueError(f"X มี {X_col.shape[0]} features แต่ layer_sizes[0] = {self.layer_sizes[0]}")
        rng = np.random.default_rng(self.seed)
        self.params_ = init_params(self.layer_sizes, self.seed, self.init_scale)
        self.optimizer.reset()
        self.history_ = {"cost": [], "val_cost": []}
        X_val_col = to_deck(X_val) if X_val is not None else None
        Y_val = self._targets(y_val) if y_val is not None else None
        best_val, patience_left, best_params = np.inf, self.early_stopping_patience, None

        for epoch in range(self.epochs):
            for X_b, Y_b in iterate_minibatches(X_col, Y, self.batch_size, rng):
                _, caches = forward(X_b, self.params_, self.activations, self.keep_prob, training=True, rng=rng)
                grads = backward(Y_b, caches, self.params_, self.activations, self.l2, self.keep_prob)
                self.optimizer.step(self.params_, grads)
            # cost ของทั้ง epoch (ไม่ใช้ dropout ตอนวัด)
            A_L, _ = forward(X_col, self.params_, self.activations)
            self.history_["cost"].append(compute_cost(A_L, Y, self.params_, self.l2))
            if X_val_col is not None:
                A_v, _ = forward(X_val_col, self.params_, self.activations)
                val_cost = compute_cost(A_v, Y_val)
                self.history_["val_cost"].append(val_cost)
                if self.early_stopping_patience is not None:
                    if val_cost < best_val - 1e-6:
                        best_val, patience_left = val_cost, self.early_stopping_patience
                        best_params = {k: v.copy() for k, v in self.params_.items()}
                    else:
                        patience_left -= 1
                        if patience_left <= 0:
                            if self.verbose:
                                print(f"early stopping ที่ epoch {epoch} (val cost ดีที่สุด {best_val:.4f})")
                            self.params_ = best_params
                            self.stopped_epoch_ = epoch
                            break
            if self.verbose and epoch % max(1, self.epochs // 10) == 0:
                msg = f"epoch {epoch:5d}  cost {self.history_['cost'][-1]:.4f}"
                if self.history_["val_cost"]:
                    msg += f"  val_cost {self.history_['val_cost'][-1]:.4f}"
                print(msg)
        return self

    # ----- การทำนาย -----------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.params_:
            raise RuntimeError("ต้อง fit ก่อน predict")
        A_L, _ = forward(to_deck(X), self.params_, self.activations)
        return A_L.reshape(-1) if A_L.shape[0] == 1 else A_L.T   # (m,) หรือ (m, C)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        P = self.predict_proba(X)
        return (P >= threshold).astype(int) if P.ndim == 1 else P.argmax(axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        y_pred = self.predict(X)
        return binary_report(y, y_pred) if self.n_classes == 2 else classification_summary(y, y_pred)

    def gradient_check(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> float:
        params = self.params_ or init_params(self.layer_sizes, self.seed, self.init_scale)
        return gradient_check(to_deck(X), self._targets(y), params, self.activations, self.l2, eps)

    # ----- บันทึก/โหลด --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "layer_sizes": self.layer_sizes, "activations": self.activations, "lr": self.lr, "epochs": self.epochs,
            "batch_size": self.batch_size, "l2": self.l2, "keep_prob": self.keep_prob, "seed": self.seed,
            "history": self.history_,
        }
        np.savez(path, meta=json.dumps(meta), **self.params_)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "NeuralNetwork":
        data = np.load(Path(path).with_suffix(".npz"))
        meta = json.loads(str(data["meta"]))
        model = cls(meta["layer_sizes"], meta["activations"], lr=meta["lr"], epochs=meta["epochs"], batch_size=meta["batch_size"], l2=meta["l2"], keep_prob=meta["keep_prob"], seed=meta["seed"])
        model.params_ = {k: data[k] for k in data.files if k != "meta"}
        model.history_ = meta["history"]
        return model

    def __repr__(self) -> str:
        return f"NeuralNetwork(layer_sizes={self.layer_sizes}, activations={self.activations}, lr={self.lr}, epochs={self.epochs}, optimizer={self.optimizer!r})"
