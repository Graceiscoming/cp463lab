"""
ฟังก์ชันวาดกราฟที่ใช้ซ้ำใน notebook และ script
Reusable matplotlib helpers

label ทุกอันเป็นภาษาอังกฤษ เพราะ font default ของ matplotlib ไม่มีตัวอักษรไทย (จะขึ้นเป็นกล่องสี่เหลี่ยม)
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np


def _plt():
    import matplotlib.pyplot as plt

    return plt


def plot_history(history: Mapping[str, Sequence[float]] | Sequence[float], ax=None, title: str = "Cost per epoch", logy: bool = False):
    """วาด cost (และ val_cost ถ้ามี) ต่อ epoch — history เป็น dict {"cost": [...], "val_cost": [...]} หรือ list"""
    plt = _plt()
    ax = ax or plt.gca()
    if isinstance(history, Mapping):
        for key, values in history.items():
            if values is not None and len(values):
                ax.plot(values, label=key)
        ax.legend()
    else:
        ax.plot(list(history), label="cost")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cost J")
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def plot_decision_boundary(predict_fn: Callable[[np.ndarray], np.ndarray], X: np.ndarray, y: np.ndarray, ax=None, title: str = "", resolution: int = 200):
    """วาดพื้นที่ที่โมเดลทายเป็น class 0/1 บนข้อมูล 2 มิติ — predict_fn รับ X (m, 2) คืน label หรือ probability (m,)"""
    plt = _plt()
    ax = ax or plt.gca()
    X = np.asarray(X)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = np.asarray(predict_fn(grid), dtype=float).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5] if zz.max() <= 1 and set(np.unique(zz)) <= {0.0, 1.0} else 20, alpha=0.3, cmap="coolwarm")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", edgecolor="k", s=25)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    return ax


def plot_confusion(cm: np.ndarray, labels: Sequence[str] | None = None, ax=None, title: str = "Confusion matrix"):
    """วาด confusion matrix (แถว = actual, คอลัมน์ = predicted) พร้อมตัวเลขในช่อง"""
    plt = _plt()
    ax = ax or plt.gca()
    cm = np.asarray(cm)
    n = cm.shape[0]
    labels = list(labels) if labels is not None else [str(i) for i in range(n)]
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(n), labels)
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(n):
        for j in range(n):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    return ax


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, auc_value: float | None = None, ax=None, label: str = "model"):
    """วาด ROC curve เทียบกับเส้นทแยง (โมเดลสุ่ม)"""
    plt = _plt()
    ax = ax or plt.gca()
    name = f"{label} (AUC = {auc_value:.3f})" if auc_value is not None else label
    ax.plot(fpr, tpr, marker=".", label=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="random")
    ax.set_xlabel("False Positive Rate (1 - specificity)")
    ax.set_ylabel("True Positive Rate (recall)")
    ax.set_title("ROC curve")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def show_images(images: np.ndarray, labels: Sequence | None = None, n: int = 10, cmap: str = "gray", title: str = ""):
    """แสดงภาพ n ภาพแรกเรียงแถวเดียว (images: (m, H, W) หรือ (m, H, W, C))"""
    plt = _plt()
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(1.4 * n, 1.8))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.imshow(images[i], cmap=cmap if images[i].ndim == 2 else None)
        ax.axis("off")
        if labels is not None:
            ax.set_title(str(labels[i]), fontsize=9)
    if title:
        fig.suptitle(title)
    return fig


def show_filters(weights: np.ndarray, n_cols: int = 8, title: str = "Learned filters", layout: str = "torch"):
    """แสดง filter ของชั้น conv แรก

    layout="torch": weights shape (n_F, C, f, f)  (ค่า default — จาก nn.Conv2d(...).weight)
    layout="deck" : weights shape (f, f, C, n_F)  (ตามสไลด์ p.191)
    ระบุ layout ให้ชัดแทนการเดาจาก shape เพราะเดาผิดได้ง่ายเมื่อ f กับ n_F ใกล้กัน
    """
    plt = _plt()
    w = np.asarray(weights)
    if layout == "deck":
        w = np.transpose(w, (3, 2, 0, 1))  # (f, f, C, n_F) → (n_F, C, f, f)
    elif layout != "torch":
        raise ValueError("layout ต้องเป็น 'torch' หรือ 'deck'")
    n_f = w.shape[0]
    n_rows = int(np.ceil(n_f / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.3 * n_rows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n_f:
            ax.imshow(w[i, 0], cmap="gray")
    fig.suptitle(title)
    return fig


def show_feature_maps(maps: np.ndarray, n_cols: int = 8, title: str = "Feature maps", layout: str = "CHW"):
    """แสดง feature map ทุก channel

    layout="CHW": maps shape (C, H, W)  (ค่า default — output ของ torch สำหรับภาพเดียว)
    layout="HWC": maps shape (H, W, C)  (ตามสไลด์ / nnlab.conv)
    ระบุ layout ให้ชัด — การเดาจาก shape ผิดได้ เช่น (16, 10, 10) จะถูกอ่านเป็น (H, W, C) ทั้งที่เป็น (C, H, W)
    """
    plt = _plt()
    m = np.asarray(maps)
    if layout == "HWC":
        m = np.transpose(m, (2, 0, 1))  # (H, W, C) → (C, H, W)
    elif layout != "CHW":
        raise ValueError("layout ต้องเป็น 'CHW' หรือ 'HWC'")
    n_c = m.shape[0]
    n_rows = int(np.ceil(n_c / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.3 * n_cols, 1.3 * n_rows))
    for i, ax in enumerate(np.atleast_1d(axes).ravel()):
        ax.axis("off")
        if i < n_c:
            ax.imshow(m[i], cmap="viridis")
    fig.suptitle(title)
    return fig
