"""
การตั้งค่าการเทรนแบบ production: dataclass เดียว ใช้ได้ทั้งจาก argparse และไฟล์ JSON
Training configuration shared by scripts and notebooks
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .utils import load_json, save_json

MODELS = ["numpy-perceptron", "numpy-mlp", "sklearn-logreg", "sklearn-mlp", "torch-logreg", "torch-mlp"]
OPTIMIZERS = ["sgd", "momentum", "rmsprop", "adam"]


@dataclass
class TrainConfig:
    """hyperparameter ทั้งหมดของการเทรนหนึ่งครั้ง (ค่า default = ค่าที่ใช้ได้กับ breast_cancer)"""

    model: str = "numpy-perceptron"
    dataset: str = "breast_cancer"
    lr: float = 0.1
    epochs: int = 300
    batch_size: int | None = None        # None = batch gradient descent (ใช้ทั้ง m)
    l2: float = 0.0                       # λ ของ L2 regularization
    keep_prob: float = 1.0                # dropout keep probability (1.0 = ไม่ใช้)
    optimizer: str = "sgd"
    hidden: tuple[int, ...] = (16,)       # จำนวน unit ของ hidden layer (ใช้กับ *-mlp)
    seed: int = 463
    out_dir: str = "runs"
    name: str | None = None               # ชื่อโฟลเดอร์ผลลัพธ์ (None = สร้างจาก timestamp)

    # ----- แปลงไปมากับ JSON / argparse -------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d

    def to_json(self, path: str | Path) -> Path:
        return save_json(self.to_dict(), path)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = {f.name for f in fields(cls)}
        clean = {k: v for k, v in d.items() if k in known}
        if "hidden" in clean and clean["hidden"] is not None:
            clean["hidden"] = tuple(int(h) for h in clean["hidden"])
        return cls(**clean)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(load_json(path))

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """เพิ่ม argument ทุกตัวให้ parser (ค่า default = None เพื่อให้รู้ว่าผู้ใช้ระบุมาหรือไม่)"""
        parser.add_argument("--config", type=str, default=None, help="ไฟล์ JSON ของ TrainConfig (argument อื่นจะ override)")
        parser.add_argument("--model", choices=MODELS, default=None)
        parser.add_argument("--dataset", type=str, default=None)
        parser.add_argument("--lr", type=float, default=None)
        parser.add_argument("--epochs", type=int, default=None)
        parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
        parser.add_argument("--l2", type=float, default=None)
        parser.add_argument("--keep-prob", dest="keep_prob", type=float, default=None)
        parser.add_argument("--optimizer", choices=OPTIMIZERS, default=None)
        parser.add_argument("--hidden", type=int, nargs="+", default=None, help="เช่น --hidden 16 8")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--out-dir", dest="out_dir", type=str, default=None)
        parser.add_argument("--name", type=str, default=None)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "TrainConfig":
        """เริ่มจาก default → ทับด้วยไฟล์ --config (ถ้ามี) → ทับด้วย argument ที่ผู้ใช้ระบุ"""
        cfg = cls.from_json(args.config) if getattr(args, "config", None) else cls()
        for f in fields(cls):
            value = getattr(args, f.name, None)
            if value is not None:
                setattr(cfg, f.name, tuple(value) if f.name == "hidden" else value)
        return cfg
