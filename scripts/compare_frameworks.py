"""
เทรน logistic regression ตัวเดียวกัน 3 วิธี (numpy จากศูนย์ / scikit-learn / PyTorch) แล้วเทียบผล
Train the same logistic regression three ways and compare weights, metrics and speed

    uv run python scripts/compare_frameworks.py --dataset breast_cancer --epochs 300 --lr 0.1
    uv run python scripts/compare_frameworks.py --mlp --hidden 16 --dataset moons      # เทียบ neural network แทน

สิ่งที่ควรเห็น: accuracy ต่างกันไม่เกิน ~0.02 และ weight ชี้ทิศทางเดียวกัน
    - numpy/torch เทรน 300 epoch ยังไม่ converge เต็มที่ → weight เล็ก (เหมือน early stopping = regularization ทางอ้อม)
    - sklearn ใช้ lbfgs จน converge + L2 (C=1.0) → weight เล็กใกล้เคียงกัน
    - ลอง --sklearn-C 1e6 (ไม่มี L2): weight จะระเบิดเป็นหลักสิบ และ accuracy ลด → เหตุผลที่ต้องมี regularization (lab08)
"""
from __future__ import annotations

import argparse

import _common  # noqa: F401
import numpy as np
from _common import print_metrics_table

from nnlab.baselines import SklearnLogReg, SklearnMLP
from nnlab.data import get_dataset
from nnlab.nn import NeuralNetwork
from nnlab.perceptron import Perceptron
from nnlab.torch_models import TorchMLP, TorchPerceptron, TorchTrainer
from nnlab.utils import Timer, set_seed, setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--mlp", action="store_true", help="เทียบ neural network (numpy-mlp / sklearn-mlp / torch-mlp)")
    parser.add_argument("--hidden", type=int, nargs="+", default=[16])
    parser.add_argument("--seed", type=int, default=463)
    parser.add_argument("--sklearn-C", dest="sklearn_C", type=float, default=1.0,
                        help="C = 1/λ ของ sklearn (default 1.0 = L2 ปกติ; ลอง 1e6 เพื่อดูว่า weight ระเบิดเมื่อไม่ regularize)")
    args = parser.parse_args()
    log = setup_logging()
    set_seed(args.seed)

    X_tr, X_te, y_tr, y_te, n_classes = get_dataset(args.dataset, args.seed)
    n_in = X_tr.shape[1]
    log.info("dataset=%s train=%s test=%s", args.dataset, X_tr.shape, X_te.shape)

    if args.mlp:
        sizes = [n_in, *args.hidden, 1]
        models = {
            "numpy-mlp": NeuralNetwork(sizes, lr=args.lr, epochs=args.epochs, seed=args.seed),
            "sklearn-mlp": SklearnMLP(hidden=tuple(args.hidden), lr=0.01, epochs=args.epochs, seed=args.seed),
            "torch-mlp": TorchTrainer(TorchMLP(sizes), loss="bce", optimizer="sgd", lr=args.lr, epochs=args.epochs, batch_size=None, seed=args.seed),
        }
    else:
        models = {
            "numpy-perceptron": Perceptron(lr=args.lr, epochs=args.epochs, seed=args.seed),
            "sklearn-logreg": SklearnLogReg(C=args.sklearn_C, max_iter=max(args.epochs, 100), seed=args.seed),
            "torch-logreg": TorchTrainer(TorchPerceptron(n_in), loss="bce", optimizer="sgd", lr=args.lr, epochs=args.epochs, batch_size=None, seed=args.seed),
        }

    rows, weights = [], {}
    for name, model in models.items():
        with Timer() as t:
            model.fit(X_tr, y_tr)
        m = model.evaluate(X_te, y_te)
        rows.append({"model": name, **m, "seconds": t.elapsed})
        if not args.mlp:
            if name.startswith("numpy"):
                weights[name] = np.r_[model.b_, model.w_.ravel()]
            elif name.startswith("sklearn"):
                weights[name] = np.r_[model.b_, model.w_.ravel()]
            else:
                lin = model.model.linear
                weights[name] = np.r_[lin.bias.detach().cpu().numpy(), lin.weight.detach().cpu().numpy().ravel()]

    print()
    print_metrics_table(rows, ["accuracy", "precision", "recall", "f1", "seconds"])
    if weights:
        print("\nparameter [b, w1, w2, w3, ...] (5 ตัวแรก)")
        for name, w in weights.items():
            print(f"  {name:<18}", np.array2string(w[:5], precision=3, suppress_small=True))
        accs = [r["accuracy"] for r in rows]
        print(f"\naccuracy ต่างกันมากสุด {max(accs) - min(accs):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
