"""
transfer learning บน data/cifar3 (bird / dog / frog) ด้วย ResNet-18: เทรนจากศูนย์ เทียบ head อย่างเดียว เทียบ fine-tune
Transfer learning CLI: scratch CNN vs frozen-backbone head vs fine-tuned ResNet-18

    uv run python scripts/train_transfer.py --mode finetune                  # freeze → เทรน head → fine-tune layer4
    uv run python scripts/train_transfer.py --mode all --fast                # เทียบสามแบบด้วยข้อมูล 100 ภาพ/class (~1 นาที CPU)
    uv run python scripts/train_transfer.py --mode scratch --epochs 30 --patience 5
    uv run python scripts/train_transfer.py --mode head --weights-path ~/resnet18-f37072fd.pth   # ไม่ต้องดาวน์โหลด

น้ำหนัก ResNet-18 (45 MB) ดาวน์โหลดอัตโนมัติครั้งแรกไปที่ torch cache (~/.cache/torch/hub/checkpoints)
ผลลัพธ์อยู่ใน runs/<timestamp>_transfer_<mode>/ : config.json, metrics.json, history.csv, confusion.png, cost.png, model.pt
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import torch
from _common import print_metrics_table
from torch import nn

from nnlab.metrics import confusion_matrix
from nnlab.torch_models import TorchTrainer, count_parameters
from nnlab.utils import Timer, ensure_dir, load_json, project_root, save_json, set_seed, setup_logging
from nnlab.vision import (
    SmallCNN, build_resnet18_transfer, extract_features, load_cifar3_loaders, param_groups, trainable_parameters, unfreeze_layer4,
)

MODES = ("scratch", "head", "finetune", "all")


def evaluate(trainer: TorchTrainer, loader, n_classes: int = 3) -> tuple[dict, np.ndarray]:
    metrics = trainer.evaluate_loader(loader)
    y_true = torch.cat([torch.as_tensor(yb) for _, yb in loader]).numpy()
    return metrics, confusion_matrix(y_true, trainer.predict_loader(loader), n_classes)


def run_scratch(cfg, loaders, log):
    train_loader, val_loader, test_loader, classes = loaders
    model = SmallCNN(n_classes=len(classes))
    trainer = TorchTrainer(model, loss="ce", optimizer="adam", lr=cfg["lr"], epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                           early_stopping_patience=cfg["patience"] or None, device=cfg["device"], seed=cfg["seed"], verbose=True, log_every=1)
    with Timer() as t:
        trainer.fit_loader(train_loader, val_loader)
    metrics, cm = evaluate(trainer, test_loader, len(classes))
    metrics.update(fit_seconds=t.elapsed, trainable_params=count_parameters(model), total_params=count_parameters(model, False),
                   stopped_epoch=trainer.stopped_epoch_, epochs_run=len(trainer.history_["cost"]))
    return model, trainer.history_, metrics, cm


def train_head(cfg, model, loaders, log):
    """ขั้น head: backbone แช่แข็ง = feature extractor → คำนวณ feature ครั้งเดียว → เทรน Linear(512, C) บน feature"""
    train_loader, val_loader, _, classes = loaders
    # feature ของ train ควรมาจากภาพที่ไม่ augment (deterministic) จึงสร้าง loader แบบ eval transform แยก
    plain_train, _, _, _ = load_cifar3_loaders(cfg["batch_size"], cfg["size"], augment=False, fast=cfg["fast"], num_workers=cfg["num_workers"], seed=cfg["seed"])
    with Timer() as t_feat:
        F_tr, y_tr = extract_features(model, plain_train, cfg["device"])
        F_va, y_va = extract_features(model, val_loader, cfg["device"])
    log.info("extract features: train %s val %s (%.1fs)", tuple(F_tr.shape), tuple(F_va.shape), t_feat.elapsed)
    head = TorchTrainer(nn.Linear(512, len(classes)), loss="ce", optimizer="adam", lr=cfg["lr"], epochs=cfg["head_epochs"], batch_size=cfg["batch_size"],
                        early_stopping_patience=cfg["patience"] or None, device=cfg["device"], seed=cfg["seed"], verbose=False)
    with Timer() as t_head:
        head.fit(F_tr.numpy(), y_tr.numpy(), F_va.numpy(), y_va.numpy())
    with torch.no_grad():                                   # ย้ายน้ำหนัก head ที่เทรนแล้วเข้าโมเดลเต็ม
        model.fc.weight.copy_(head.model.weight.cpu())
        model.fc.bias.copy_(head.model.bias.cpu())
    log.info("head: %d epoch (%.1fs) val acc %.3f", len(head.history_["cost"]), t_head.elapsed, head.history_["val_acc"][-1])
    return head.history_, t_feat.elapsed + t_head.elapsed, head.stopped_epoch_


def run_head(cfg, loaders, log):
    _, _, test_loader, classes = loaders
    model = build_resnet18_transfer(len(classes), freeze=True, weights=cfg["weights"]).to(cfg["device"])
    history, seconds, stopped = train_head(cfg, model, loaders, log)
    trainer = TorchTrainer(model, loss="ce", device=cfg["device"])
    metrics, cm = evaluate(trainer, test_loader, len(classes))
    metrics.update(fit_seconds=seconds, trainable_params=count_parameters(model), total_params=count_parameters(model, False), stopped_epoch=stopped, epochs_run=len(history["cost"]))
    return model, history, metrics, cm


def run_finetune(cfg, loaders, log):
    train_loader, val_loader, test_loader, classes = loaders
    model = build_resnet18_transfer(len(classes), freeze=True, weights=cfg["weights"]).to(cfg["device"])
    _, head_seconds, _ = train_head(cfg, model, loaders, log)
    unfreeze_layer4(model)                                    # ปลด block สุดท้าย + head
    opt = torch.optim.Adam(param_groups(model, lr_backbone=cfg["lr_backbone"], lr_head=cfg["lr"]))
    trainer = TorchTrainer(model, loss="ce", optimizer=opt, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
                           early_stopping_patience=cfg["patience"] or None, device=cfg["device"], seed=cfg["seed"], verbose=True, log_every=1)
    with Timer() as t:
        trainer.fit_loader(train_loader, val_loader)
    metrics, cm = evaluate(trainer, test_loader, len(classes))
    metrics.update(fit_seconds=head_seconds + t.elapsed, trainable_params=count_parameters(model), total_params=count_parameters(model, False),
                   stopped_epoch=trainer.stopped_epoch_, epochs_run=len(trainer.history_["cost"]))
    return model, trainer.history_, metrics, cm


def save_run(cfg, mode, model, history, metrics, cm, classes, log) -> Path:
    out = ensure_dir(Path(project_root() / cfg["out_dir"]) / ((cfg["name"] + f"_{mode}") if cfg["name"] else f"{time.strftime('%Y%m%d-%H%M%S')}_transfer_{mode}"))
    save_json({**cfg, "mode": mode, "weights": str(cfg["weights"])}, out / "config.json")
    save_json(metrics, out / "metrics.json")
    torch.save({"state_dict": model.state_dict(), "classes": classes, "size": cfg["size"], "mode": mode}, out / "model.pt")
    np.savetxt(out / "confusion.csv", cm, fmt="%d", delimiter=",")
    keys = [k for k, v in history.items() if v]
    with open(out / "history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", *keys])
        for i in range(max(len(history[k]) for k in keys)):
            w.writerow([i, *[history[k][i] if i < len(history[k]) else "" for k in keys]])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from nnlab.plotting import plot_confusion, plot_history

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_history({k: history[k] for k in keys if k != "val_acc"}, ax=ax, title=f"{mode}: cost per epoch")
    fig.tight_layout(); fig.savefig(out / "cost.png", dpi=120); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    plot_confusion(cm, labels=classes, ax=ax, title=f"{mode}: test confusion")
    fig.tight_layout(); fig.savefig(out / "confusion.png", dpi=120); plt.close(fig)
    log.info("บันทึกผลที่ %s", out)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--mode", choices=MODES, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="epoch ของขั้นหลัก (scratch หรือ fine-tune)")
    parser.add_argument("--head-epochs", dest="head_epochs", type=int, default=None)
    parser.add_argument("--size", type=int, default=None, help="ขนาดภาพหลัง resize (32 = ต้นฉบับ, 64 = แนะนำสำหรับ ResNet)")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="learning rate ของ head / scratch")
    parser.add_argument("--lr-backbone", dest="lr_backbone", type=float, default=None, help="learning rate ของ layer4 ตอน fine-tune")
    parser.add_argument("--patience", type=int, default=None, help="early stopping (0 = ปิด)")
    parser.add_argument("--fast", action="store_true", default=None, help="ใช้ train 100 ภาพ/class")
    parser.add_argument("--no-augment", dest="augment", action="store_false", default=None)
    parser.add_argument("--weights-path", dest="weights_path", type=Path, default=None, help="ไฟล์ state_dict ของ resnet18 (ไม่ต้องดาวน์โหลด)")
    parser.add_argument("--no-pretrained", action="store_true", default=None, help="ResNet-18 สุ่ม (สำหรับทดสอบ)")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-dir", dest="out_dir", default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    log = setup_logging()

    cfg = {"mode": "finetune", "epochs": 8, "head_epochs": 30, "size": 64, "batch_size": 64, "lr": 1e-3, "lr_backbone": 1e-4,
           "patience": 3, "fast": False, "augment": True, "weights": "DEFAULT", "num_workers": 0, "device": "cpu", "seed": 463,
           "out_dir": "runs", "name": None}
    if args.config:
        cfg.update(load_json(args.config))
    cfg.update({k: v for k, v in vars(args).items() if k not in ("config", "weights_path", "no_pretrained") and v is not None})
    if args.weights_path:
        cfg["weights"] = str(args.weights_path.expanduser())
    if args.no_pretrained:
        cfg["weights"] = None

    set_seed(cfg["seed"])
    loaders = load_cifar3_loaders(cfg["batch_size"], cfg["size"], augment=cfg["augment"], fast=cfg["fast"], num_workers=cfg["num_workers"], seed=cfg["seed"])
    classes = loaders[3]
    log.info("cifar3: train %d | val %d | test %d | classes %s | size %d | device %s", len(loaders[0].dataset), len(loaders[1].dataset), len(loaders[2].dataset), classes, cfg["size"], cfg["device"])

    runners = {"scratch": run_scratch, "head": run_head, "finetune": run_finetune}
    modes = list(runners) if cfg["mode"] == "all" else [cfg["mode"]]
    rows = []
    for mode in modes:
        log.info("===== %s =====", mode)
        model, history, metrics, cm = runners[mode](cfg, loaders, log)
        save_run(cfg, mode, model, history, metrics, cm, classes, log)
        rows.append({"model": mode, **metrics})
    print()
    print_metrics_table(rows, ["accuracy", "macro_f1", "trainable_params", "total_params", "epochs_run", "fit_seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
