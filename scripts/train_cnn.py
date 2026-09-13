"""
เทรน LeNet-5 บน MNIST (หรือ sklearn digits แบบ offline) ด้วย PyTorch แล้วบันทึกโมเดล/รูป filter
Train LeNet-5 on MNIST (or the offline sklearn digits fallback)

    uv run python scripts/train_cnn.py --epochs 1 --limit-train 10000        # ~1-3 นาทีบน CPU
    uv run python scripts/train_cnn.py --epochs 3 --augment                  # เต็ม 60,000 ภาพ + data augmentation
    uv run python scripts/train_cnn.py --dataset digits --epochs 20          # ไม่ต้องต่ออินเทอร์เน็ต (ภาพ 8×8 ขยายเป็น 28×28)
    uv run python scripts/train_cnn.py --config configs/lenet5_mnist.json
    uv run python scripts/train_cnn.py --epochs 20 --limit-train 5000 --patience 2    # early stopping บน val 10% ของ train

ผลลัพธ์อยู่ใน runs/<timestamp>_lenet5_<dataset>/ : model.pt, metrics.json, history.csv, confusion.png, filters.png
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from _common import print_metrics_table

from nnlab.metrics import confusion_matrix
from nnlab.torch_models import LeNet5, TorchTrainer, count_parameters
from nnlab.utils import Timer, ensure_dir, load_json, project_root, save_json, set_seed, setup_logging


def load_images(dataset: str, limit_train: int | None, log):
    """คืน X_train, y_train, X_test, y_test เป็น float32 (m, 1, 28, 28) ช่วง [0, 1]"""
    if dataset == "mnist":
        from nnlab.data import load_mnist

        Xtr, ytr = load_mnist(train=True, limit=limit_train)
        Xte, yte = load_mnist(train=False)
    else:  # digits 8×8 → ขยายเป็น 28×28 ด้วย nearest neighbour เพื่อใช้ architecture เดียวกัน
        from nnlab.data import load_digits_images, stratified_split

        images, y = load_digits_images()
        images = (images / 16.0 * 255).astype(np.uint8)
        images = np.repeat(np.repeat(images, 4, axis=1), 4, axis=2)          # 8×8 → 32×32
        images = images[:, 2:30, 2:30]                                        # ตัดเป็น 28×28
        Xtr, Xte, ytr, yte = stratified_split(images, y, 0.2)
    to_float = lambda a: (a.astype(np.float32) / 255.0)[:, None, :, :]      # (m, 28, 28) → (m, 1, 28, 28)
    log.info("%s: train %d ภาพ, test %d ภาพ", dataset, len(ytr), len(yte))
    return to_float(Xtr), ytr, to_float(Xte), yte


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, augment: bool, seed: int):
    """DataLoader จาก numpy; ถ้า augment ใช้ transforms หมุน/เลื่อนภาพแบบสุ่มทุก epoch (สไลด์ p.208-210)"""
    from torchvision import transforms

    X_t = torch.as_tensor(X)
    y_t = torch.as_tensor(y, dtype=torch.long)
    if augment:
        aug = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])

        class AugDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(y_t)

            def __getitem__(self, i):
                return aug(X_t[i]), y_t[i]

        ds = AugDataset()
    else:
        ds = torch.utils.data.TensorDataset(X_t, y_t)
    gen = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None, help="ไฟล์ JSON (argument อื่นจะ override)")
    parser.add_argument("--dataset", choices=["mnist", "digits"], default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--limit-train", dest="limit_train", type=int, default=None, help="ใช้ภาพ train แค่ N ภาพ (เร็วขึ้น)")
    parser.add_argument("--augment", action="store_true", default=None)
    parser.add_argument("--pool", choices=["max", "avg"], default=None)
    parser.add_argument("--patience", type=int, default=None, help="early stopping: หยุดเมื่อ val cost ไม่ดีขึ้นติดกัน N epoch (0 = ปิด)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", dest="out_dir", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    log = setup_logging()

    cfg = {"dataset": "mnist", "epochs": 1, "batch_size": 64, "lr": 1e-3, "limit_train": None, "augment": False,
           "pool": "max", "patience": 0, "seed": 463, "out_dir": "runs", "name": None}
    if args.config:
        cfg.update(load_json(args.config))
    cfg.update({k: v for k, v in vars(args).items() if k != "config" and v is not None})

    set_seed(cfg["seed"])
    Xtr, ytr, Xte, yte = load_images(cfg["dataset"], cfg["limit_train"], log)
    # แบ่ง val 10% จาก train สำหรับดู curve / early stopping — test ต้องไม่ถูกใช้ตัดสินใจระหว่างเทรน (leakage)
    from nnlab.data import stratified_split

    Xtr, Xva, ytr, yva = stratified_split(Xtr, ytr, 0.1, cfg["seed"])
    log.info("แบ่ง val จาก train: train %d | val %d | test %d", len(ytr), len(yva), len(yte))
    train_loader = make_loader(Xtr, ytr, cfg["batch_size"], True, cfg["augment"], cfg["seed"])
    val_loader = make_loader(Xva, yva, 512, False, False, cfg["seed"])
    test_loader = make_loader(Xte, yte, 512, False, False, cfg["seed"])

    model = LeNet5(n_classes=10, in_channels=1, pad=2, pool=cfg["pool"])
    log.info("LeNet5 parameters = %s", f"{count_parameters(model):,}")
    for name, shape in model.shapes((1, 28, 28)):
        log.info("  %-12s %s", name, "×".join(map(str, shape)))
    trainer = TorchTrainer(model, loss="ce", optimizer="adam", lr=cfg["lr"], epochs=cfg["epochs"], batch_size=cfg["batch_size"], seed=cfg["seed"], verbose=True, log_every=1,
                           early_stopping_patience=cfg["patience"] or None)
    with Timer() as t:
        trainer.fit_loader(train_loader, val_loader)
    metrics = trainer.evaluate(Xte, yte)
    metrics["fit_seconds"] = t.elapsed
    metrics["n_train"] = int(len(ytr))
    metrics["epochs_run"] = len(trainer.history_["cost"])
    metrics["stopped_epoch"] = trainer.stopped_epoch_
    metrics["best_epoch"] = trainer.best_epoch_

    out = ensure_dir(Path(project_root() / cfg["out_dir"]) / (cfg["name"] or f"{time.strftime('%Y%m%d-%H%M%S')}_lenet5_{cfg['dataset']}"))
    save_json(cfg, out / "config.json")
    save_json(metrics, out / "metrics.json")
    trainer.save(out / "model")
    cm = confusion_matrix(yte, trainer.predict(Xte), 10)
    np.savetxt(out / "confusion.csv", cm, fmt="%d", delimiter=",")
    import csv

    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "cost", "val_cost", "val_acc"])
        for i, (c, vc, va) in enumerate(zip(trainer.history_["cost"], trainer.history_["val_cost"], trainer.history_["val_acc"])):
            w.writerow([i, c, vc, va])

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nnlab.plotting import plot_confusion, plot_history, show_filters

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_history({"cost": trainer.history_["cost"], "val_cost": trainer.history_["val_cost"]}, ax=ax)
    fig.tight_layout(); fig.savefig(out / "cost.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    plot_confusion(cm, ax=ax)
    fig.tight_layout(); fig.savefig(out / "confusion.png", dpi=120); plt.close(fig)
    fig = show_filters(model.features[0].weight.detach().cpu().numpy(), n_cols=6, title="CONV1 filters (6 × 5×5)")
    fig.savefig(out / "filters.png", dpi=120); plt.close(fig)

    print()
    print_metrics_table([{"model": "lenet5", **metrics}], ["accuracy", "macro_f1", "fit_seconds", "n_train", "epochs_run"])
    print(f"\nผลลัพธ์ทั้งหมดอยู่ที่: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
