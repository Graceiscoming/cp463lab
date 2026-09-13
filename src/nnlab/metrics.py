"""
การประเมินผล classification (สไลด์ p.126-152)
Classification metrics built from the confusion matrix

ทุกอย่างเริ่มจาก confusion matrix: แถว = ค่าจริง (actual), คอลัมน์ = ค่าที่ทำนาย (predict)
เหมือน sklearn.metrics.confusion_matrix เพื่อให้เทียบกันได้ตรงๆ

binary (class 1 = positive):
    cm = [[TN, FP],
          [FN, TP]]
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """นับจำนวน sample ในแต่ละช่อง (actual, predict) → matrix (C, C) ชนิด int

    >>> confusion_matrix([1,1,0,0,1], [1,0,0,1,1])
    array([[1, 1],
           [1, 2]])
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    y_pred = np.asarray(y_pred).astype(int).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} กับ y_pred {y_pred.shape} ต้องยาวเท่ากัน")
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def binary_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """TP, FP, TN, FN ของ binary classification (positive = 1)   (สไลด์ p.127-131)"""
    cm = confusion_matrix(y_true, y_pred, n_classes=2)
    return {"tp": int(cm[1, 1]), "fp": int(cm[0, 1]), "tn": int(cm[0, 0]), "fn": int(cm[1, 0])}


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """(TP + TN) / ทั้งหมด   — ใช้ได้ทั้ง binary และ multiclass"""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean(y_true == y_pred))


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FP)   — ในที่ทำนายว่า positive มีจริงกี่ส่วน (p.133)"""
    c = binary_counts(y_true, y_pred)
    return _safe_div(c["tp"], c["tp"] + c["fp"])


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FN)   — positive จริงทั้งหมด จับได้กี่ส่วน (sensitivity, p.134)"""
    c = binary_counts(y_true, y_pred)
    return _safe_div(c["tp"], c["tp"] + c["fn"])


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TN / (TN + FP)   — negative จริงทั้งหมด ทายถูกกี่ส่วน (p.135)"""
    c = binary_counts(y_true, y_pred)
    return _safe_div(c["tn"], c["tn"] + c["fp"])


def f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """harmonic mean ของ precision กับ recall = 2PR / (P + R)   (p.136)"""
    p, r = precision(y_true, y_pred), recall(y_true, y_pred)
    return _safe_div(2 * p * r, p + r)


def binary_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """รวมทุก metric ของ binary classification ไว้ใน dict เดียว"""
    c = binary_counts(y_true, y_pred)
    p = _safe_div(c["tp"], c["tp"] + c["fp"])
    r = _safe_div(c["tp"], c["tp"] + c["fn"])
    return {
        **c,
        "accuracy": _safe_div(c["tp"] + c["tn"], sum(c.values())),
        "precision": p,
        "recall": r,
        "specificity": _safe_div(c["tn"], c["tn"] + c["fp"]),
        "f1": _safe_div(2 * p * r, p + r),
    }


# ---------------------------------------------------------------------------
# ROC (สไลด์ p.140-141)
# ---------------------------------------------------------------------------
def roc_curve(y_true: np.ndarray, scores: np.ndarray, thresholds: np.ndarray | None = None):
    """กวาด threshold τ จาก 1 → 0 แล้วเก็บ (FPR, TPR) แต่ละจุด

    τ = 1: ทุกตัวถูกทายเป็น negative → (FPR, TPR) = (0, 0)
    τ = 0: ทุกตัวถูกทายเป็น positive → (1, 1)
    คืน fpr, tpr, thresholds (เรียงให้ fpr เพิ่มขึ้น เพื่อคำนวณ AUC ได้)
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    scores = np.asarray(scores, dtype=float).reshape(-1)
    if thresholds is None:
        thresholds = np.concatenate([[np.inf], np.unique(scores)[::-1], [-np.inf]])
    n_pos = max(int((y_true == 1).sum()), 1)
    n_neg = max(int((y_true == 0).sum()), 1)
    tpr, fpr = [], []
    for tau in thresholds:
        pred = (scores >= tau).astype(int)
        tpr.append(np.sum((pred == 1) & (y_true == 1)) / n_pos)
        fpr.append(np.sum((pred == 1) & (y_true == 0)) / n_neg)
    return np.array(fpr), np.array(tpr), np.asarray(thresholds)


def auc(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """พื้นที่ใต้ ROC ด้วยกฎสี่เหลี่ยมคางหมู (trapezoid) — 1.0 = สมบูรณ์แบบ, 0.5 = สุ่ม"""
    order = np.argsort(fpr, kind="stable")
    x, y = np.asarray(fpr)[order], np.asarray(tpr)[order]
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))


# ---------------------------------------------------------------------------
# multiclass: micro / macro average (สไลด์ p.142-151)
# ---------------------------------------------------------------------------
def per_class_counts(cm: np.ndarray) -> dict[str, np.ndarray]:
    """แยก TP/FP/FN/TN ของแต่ละ class จาก confusion matrix (C, C) แบบ one-vs-rest

    TP_c = ช่องทแยง, FP_c = คอลัมน์ c ลบ TP, FN_c = แถว c ลบ TP, TN_c = ที่เหลือ
    """
    cm = np.asarray(cm)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def multiclass_report(cm: np.ndarray, average: str = "macro") -> dict[str, float]:
    """precision / recall / f1 ของ multiclass

    average="micro": รวม TP, FP, FN ของทุก class ก่อน แล้วค่อยคำนวณ (p.144-147)
                     → micro precision = micro recall = micro f1 = accuracy
    average="macro": คำนวณ metric ต่อ class ก่อน แล้วเฉลี่ยเท่ากันทุก class (p.148-151)
                     → class เล็กมีน้ำหนักเท่า class ใหญ่
    accuracy ไม่มี micro/macro (มีค่าเดียว)
    """
    c = per_class_counts(cm)
    tp, fp, fn = c["tp"], c["fp"], c["fn"]
    acc = _safe_div(tp.sum(), np.asarray(cm).sum())
    if average == "micro":
        p = _safe_div(tp.sum(), tp.sum() + fp.sum())
        r = _safe_div(tp.sum(), tp.sum() + fn.sum())
        f = _safe_div(2 * p * r, p + r)
    elif average == "macro":
        with np.errstate(divide="ignore", invalid="ignore"):
            p_c = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
            r_c = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
            f_c = np.where(p_c + r_c > 0, 2 * p_c * r_c / (p_c + r_c), 0.0)
        p, r, f = float(p_c.mean()), float(r_c.mean()), float(f_c.mean())
    else:
        raise ValueError("average ต้องเป็น 'micro' หรือ 'macro'")
    return {"accuracy": acc, "precision": float(p), "recall": float(r), "f1": float(f)}


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """เลือกอัตโนมัติ: binary → binary_report, multiclass → accuracy + macro/micro"""
    n_classes = int(max(np.max(y_true), np.max(y_pred))) + 1
    if n_classes <= 2:
        return binary_report(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    macro = multiclass_report(cm, "macro")
    micro = multiclass_report(cm, "micro")
    return {
        "accuracy": macro["accuracy"],
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "micro_f1": micro["f1"],
    }
