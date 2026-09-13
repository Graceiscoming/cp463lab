"""
รันการทดลองหนึ่งครั้งแบบ production: โหลดข้อมูล → สร้างโมเดล → เทรน → ประเมิน → บันทึกทุกอย่างลง runs/
One reproducible experiment: data → model → fit → evaluate → artifacts on disk

ทุก run สร้างโฟลเดอร์  runs/<timestamp>_<model>_<dataset>/  ที่มี
    config.json     TrainConfig ที่ใช้ (ทำซ้ำได้)
    metrics.json    metrics บน test set
    history.csv     cost ต่อ epoch
    model.*         น้ำหนักโมเดล (.npz / .joblib / .pt)
    cost.png, confusion.png
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from .config import TrainConfig
from .data import get_dataset
from .metrics import confusion_matrix
from .utils import Timer, ensure_dir, get_logger, load_json, project_root, save_json, set_seed


def build_model(cfg: TrainConfig, n_in: int, n_classes: int):
    """สร้างโมเดลตามชื่อใน cfg.model — ทุกตัวมี interface เดียวกัน (fit/predict/evaluate/save)"""
    n_out = 1 if n_classes == 2 else n_classes
    if cfg.model == "numpy-perceptron":
        from .perceptron import Perceptron

        if n_classes != 2:
            raise ValueError("numpy-perceptron รองรับเฉพาะ binary classification — ใช้ numpy-mlp สำหรับ multiclass")
        return Perceptron(lr=cfg.lr, epochs=cfg.epochs, l2=cfg.l2, seed=cfg.seed)
    if cfg.model == "numpy-mlp":
        from .nn import NeuralNetwork

        return NeuralNetwork([n_in, *cfg.hidden, n_out], lr=cfg.lr, epochs=cfg.epochs, batch_size=cfg.batch_size,
                             l2=cfg.l2, keep_prob=cfg.keep_prob, optimizer=cfg.optimizer, seed=cfg.seed)
    if cfg.model == "sklearn-logreg":
        from .baselines import SklearnLogReg

        C = 1.0 / cfg.l2 if cfg.l2 > 0 else 1e6
        return SklearnLogReg(C=C, max_iter=max(cfg.epochs, 100), seed=cfg.seed)
    if cfg.model == "sklearn-mlp":
        from .baselines import SklearnMLP

        return SklearnMLP(hidden=cfg.hidden, lr=cfg.lr, epochs=cfg.epochs, alpha=cfg.l2, seed=cfg.seed)
    if cfg.model in ("torch-logreg", "torch-mlp"):
        from .torch_models import TorchMLP, TorchPerceptron, TorchTrainer

        if cfg.model == "torch-logreg":
            if n_classes != 2:
                raise ValueError("torch-logreg รองรับเฉพาะ binary classification")
            net = TorchPerceptron(n_in)
        else:
            net = TorchMLP([n_in, *cfg.hidden, n_out], p_drop=1.0 - cfg.keep_prob)
        return TorchTrainer(net, loss="bce" if n_classes == 2 else "ce", optimizer=cfg.optimizer, lr=cfg.lr,
                            epochs=cfg.epochs, batch_size=cfg.batch_size, weight_decay=cfg.l2, seed=cfg.seed)
    raise ValueError(f"ไม่รู้จัก model '{cfg.model}'")


def run_dir_for(cfg: TrainConfig) -> Path:
    base = Path(cfg.out_dir)
    if not base.is_absolute():
        base = project_root() / base
    name = cfg.name or f"{time.strftime('%Y%m%d-%H%M%S')}_{cfg.model}_{cfg.dataset}"
    return ensure_dir(base / name)


def run(cfg: TrainConfig, save_plots: bool = True) -> Path:
    """เทรนและประเมินตาม cfg แล้วคืน path ของโฟลเดอร์ผลลัพธ์"""
    log = get_logger()
    set_seed(cfg.seed)
    X_tr, X_te, y_tr, y_te, n_classes = get_dataset(cfg.dataset, cfg.seed)
    model = build_model(cfg, X_tr.shape[1], n_classes)
    log.info("dataset=%s  train=%s  test=%s  n_classes=%d", cfg.dataset, X_tr.shape, X_te.shape, n_classes)
    log.info("model=%r", model)

    with Timer() as t:
        model.fit(X_tr, y_tr)
    metrics = model.evaluate(X_te, y_te)
    metrics["train_accuracy"] = float(np.mean(model.predict(X_tr) == y_tr))
    metrics["fit_seconds"] = t.elapsed
    log.info("test accuracy=%.4f  (fit %.2fs)", metrics["accuracy"], t.elapsed)

    out = run_dir_for(cfg)
    cfg.to_json(out / "config.json")
    save_json(metrics, out / "metrics.json")
    model.save(out / "model")
    _write_history(model.history_, out / "history.csv")
    cm = confusion_matrix(y_te, model.predict(X_te), n_classes)
    np.savetxt(out / "confusion.csv", cm, fmt="%d", delimiter=",")
    if save_plots:
        _save_plots(model.history_, cm, out)
    log.info("บันทึกผลที่ %s", out)
    return out


def _write_history(history: dict[str, list[float]], path: Path) -> None:
    keys = [k for k, v in history.items() if v]
    if not keys:
        return
    n = max(len(history[k]) for k in keys)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", *keys])
        for i in range(n):
            w.writerow([i, *[history[k][i] if i < len(history[k]) else "" for k in keys]])


def _save_plots(history: dict, cm: np.ndarray, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .plotting import plot_confusion, plot_history

    if any(history.values()):
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_history({k: v for k, v in history.items() if v}, ax=ax)
        fig.tight_layout()
        fig.savefig(out / "cost.png", dpi=120)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    plot_confusion(cm, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "confusion.png", dpi=120)
    plt.close(fig)


def load_run(run_dir: str | Path):
    """โหลด config + โมเดลจากโฟลเดอร์ผลลัพธ์ (สร้างโมเดลใหม่ตาม config แล้วโหลดน้ำหนัก)"""
    run_dir = Path(run_dir)
    cfg = TrainConfig.from_json(run_dir / "config.json")
    X_tr, X_te, y_tr, y_te, n_classes = get_dataset(cfg.dataset, cfg.seed)
    model = build_model(cfg, X_tr.shape[1], n_classes)
    model_path = next(run_dir.glob("model.*"))
    if hasattr(type(model), "load") and cfg.model.startswith(("numpy", "sklearn")):
        model = type(model).load(model_path)
    else:
        model.load(model_path)
    metrics = load_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    return cfg, model, (X_tr, X_te, y_tr, y_te), metrics


def latest_run(out_dir: str | Path = "runs") -> Path:
    base = Path(out_dir) if Path(out_dir).is_absolute() else project_root() / out_dir
    runs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "config.json").exists())
    if not runs:
        raise FileNotFoundError(f"ไม่มี run ใน {base}")
    return runs[-1]
