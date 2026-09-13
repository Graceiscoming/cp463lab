"""
ประเมินโมเดลที่เทรนไว้แล้วจากโฟลเดอร์ runs/<run>/ : confusion matrix, metrics, ROC (binary)
Evaluate a saved run: confusion matrix, metrics table, ROC curve

    uv run python scripts/evaluate.py --latest
    uv run python scripts/evaluate.py --run runs/20260907-120000_numpy-perceptron_breast_cancer --threshold 0.3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _common  # noqa: F401
import numpy as np
from _common import print_metrics_table

from nnlab.experiment import latest_run, load_run
from nnlab.metrics import auc, binary_report, classification_summary, confusion_matrix, roc_curve
from nnlab.utils import save_json, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, default=None, help="โฟลเดอร์ผลลัพธ์")
    parser.add_argument("--latest", action="store_true", help="ใช้ run ล่าสุดใน runs/")
    parser.add_argument("--threshold", type=float, default=0.5, help="τ สำหรับตัดสิน class 1 (binary)")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    log = setup_logging()

    run_dir = latest_run() if args.latest or args.run is None else args.run
    cfg, model, (X_tr, X_te, y_tr, y_te), saved = load_run(run_dir)
    log.info("โหลด %s (model=%s dataset=%s)", run_dir.name, cfg.model, cfg.dataset)

    P = model.predict_proba(X_te)
    binary = P.ndim == 1
    y_pred = (P >= args.threshold).astype(int) if binary else P.argmax(axis=1)
    n_classes = 2 if binary else P.shape[1]
    cm = confusion_matrix(y_te, y_pred, n_classes)
    metrics = binary_report(y_te, y_pred) if binary else classification_summary(y_te, y_pred)

    print(f"\nConfusion matrix (แถว = actual, คอลัมน์ = predicted) threshold={args.threshold if binary else '-'}")
    print(cm)
    print()
    cols = [c for c in ["accuracy", "precision", "recall", "specificity", "f1", "macro_f1", "micro_f1"] if c in metrics]
    print_metrics_table([{"model": cfg.model, **metrics}], cols)

    if binary:
        fpr, tpr, _ = roc_curve(y_te, P)
        metrics["auc"] = auc(fpr, tpr)
        print(f"\nAUC = {metrics['auc']:.4f}")
        if not args.no_plots:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            from nnlab.plotting import plot_roc

            fig, ax = plt.subplots(figsize=(4.5, 4))
            plot_roc(fpr, tpr, metrics["auc"], ax=ax, label=cfg.model)
            fig.tight_layout()
            fig.savefig(run_dir / "roc.png", dpi=120)
            log.info("บันทึก %s", run_dir / "roc.png")
    save_json({**metrics, "threshold": args.threshold}, run_dir / f"metrics_eval_t{args.threshold:.2f}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
