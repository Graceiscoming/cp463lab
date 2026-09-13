"""
เทรนโมเดล classification หนึ่งตัวจาก command line แล้วบันทึกผลลง runs/
Train one classifier from the CLI and persist everything under runs/

ตัวอย่าง
    uv run python scripts/train.py --model numpy-perceptron --dataset breast_cancer --epochs 300 --lr 0.1
    uv run python scripts/train.py --model numpy-mlp --dataset moons --hidden 16 8 --optimizer adam --lr 0.01 --epochs 500
    uv run python scripts/train.py --model torch-mlp --dataset digits --hidden 64 --batch-size 64 --epochs 30
    uv run python scripts/train.py --config configs/example_train.json --epochs 50      # ไฟล์ config + override

โมเดล   : numpy-perceptron | numpy-mlp | sklearn-logreg | sklearn-mlp | torch-logreg | torch-mlp
dataset : lung | churn | breast_cancer | moons | blobs | digits
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401  (ตั้ง sys.path + UTF-8)
from _common import print_metrics_table

from nnlab.config import TrainConfig
from nnlab.experiment import run
from nnlab.utils import load_json, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    TrainConfig.add_arguments(parser)
    parser.add_argument("--no-plots", action="store_true", help="ไม่บันทึกรูป (เร็วขึ้นเล็กน้อย)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)

    cfg = TrainConfig.from_args(args)
    out = run(cfg, save_plots=not args.no_plots)
    metrics = load_json(out / "metrics.json")
    cols = [c for c in ["accuracy", "precision", "recall", "f1", "macro_f1", "train_accuracy", "fit_seconds"] if c in metrics]
    print()
    print_metrics_table([{"model": cfg.model, **metrics}], cols)
    print(f"\nผลลัพธ์ทั้งหมดอยู่ที่: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
