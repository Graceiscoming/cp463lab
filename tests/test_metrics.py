import numpy as np
import pytest
from sklearn import metrics as skm

from nnlab.metrics import (
    accuracy, auc, binary_counts, binary_report, confusion_matrix, f1, multiclass_report, per_class_counts,
    precision, recall, roc_curve, specificity,
)

# ตารางสไลด์ p.131: y จริง กับ ŷ ที่ทำนาย (10 ตัว) → TP=4, TN=3, FP=2, FN=1
Y_P131 = np.array([1, 1, 1, 0, 0, 0, 1, 0, 0, 1])
YHAT_P131 = np.array([1, 0, 1, 0, 1, 0, 1, 1, 0, 1])


def test_confusion_matrix_layout_matches_sklearn(rng):
    y = rng.integers(0, 3, 60)
    p = rng.integers(0, 3, 60)
    assert np.array_equal(confusion_matrix(y, p), skm.confusion_matrix(y, p))


def test_binary_counts_slide_p131():
    c = binary_counts(Y_P131, YHAT_P131)
    assert (c["tp"], c["tn"], c["fp"], c["fn"]) == (4, 3, 2, 1)


def test_binary_metrics_match_sklearn(rng):
    y = rng.integers(0, 2, 200)
    p = (rng.random(200) < 0.6).astype(int)
    assert accuracy(y, p) == pytest.approx(skm.accuracy_score(y, p))
    assert precision(y, p) == pytest.approx(skm.precision_score(y, p))
    assert recall(y, p) == pytest.approx(skm.recall_score(y, p))
    assert f1(y, p) == pytest.approx(skm.f1_score(y, p))
    tn, fp, _, _ = skm.confusion_matrix(y, p).ravel()
    assert specificity(y, p) == pytest.approx(tn / (tn + fp))
    rep = binary_report(y, p)
    assert rep["f1"] == pytest.approx(skm.f1_score(y, p))


def test_roc_and_auc_match_sklearn(rng):
    y = rng.integers(0, 2, 100)
    s = np.clip(y * 0.4 + rng.random(100) * 0.6, 0, 1)
    fpr, tpr, _ = roc_curve(y, s)
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1
    assert auc(fpr, tpr) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-9)


def test_multiclass_micro_macro_slide_p144_151():
    """confusion matrix ของ Virus/Bacteria/Fungus จากสไลด์ (แถว = actual, คอลัมน์ = predicted)"""
    cm = np.array([[16, 0, 5], [0, 14, 0], [1, 1, 6]])
    c = per_class_counts(cm)
    assert c["tp"].tolist() == [16, 14, 6]
    assert c["fp"].tolist() == [1, 1, 5]
    assert c["fn"].tolist() == [5, 0, 2]
    assert c["tn"].tolist() == [21, 28, 30]
    micro = multiclass_report(cm, "micro")
    assert micro["precision"] == pytest.approx(36 / 43) and micro["recall"] == pytest.approx(36 / 43)
    assert micro["f1"] == pytest.approx(36 / 43) and micro["accuracy"] == pytest.approx(36 / 43)
    macro = multiclass_report(cm, "macro")           # errata: สไลด์ใช้สูตร micro ซ้ำในหน้า macro
    assert macro["precision"] == pytest.approx(np.mean([16 / 17, 14 / 15, 6 / 11]), abs=1e-9)
    assert macro["recall"] == pytest.approx(np.mean([16 / 21, 1.0, 6 / 8]), abs=1e-9)
    assert macro["f1"] == pytest.approx(0.8133, abs=1e-3)
    # เทียบ sklearn จาก label ที่สร้างจาก cm
    y_true = np.repeat(np.arange(3), cm.sum(axis=1))
    y_pred = np.concatenate([np.repeat(np.arange(3), cm[i]) for i in range(3)])
    assert macro["f1"] == pytest.approx(skm.f1_score(y_true, y_pred, average="macro"))
    assert micro["f1"] == pytest.approx(skm.f1_score(y_true, y_pred, average="micro"))
