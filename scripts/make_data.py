"""
สร้างไฟล์ข้อมูลทั้งหมดใน data/ แบบ deterministic (รันกี่ครั้งก็ได้ไฟล์เดิม byte-identical)
Generate every CSV in data/ deterministically

    uv run python scripts/make_data.py           # สร้าง/เขียนทับ
    uv run python scripts/make_data.py --check   # ตรวจว่าไฟล์ที่มีอยู่ตรงกับที่สร้างใหม่หรือไม่ (ใช้ใน test)

ไฟล์ที่สร้าง
    lung_cancer_toy.csv   3 แถวจากสไลด์ p.40  (x1 smoking/week, x2 chest pain, y lung cancer)
    churn_toy.csv         5 แถวจากสไลด์ p.72  (monthly usage, subscription type, churn Yes/No)
    churn_synthetic.csv   1,000 แถว สังเคราะห์จาก logistic ground truth (churn ≈ 30%)
"""
from __future__ import annotations

import argparse
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nnlab.activations import sigmoid  # noqa: E402
from nnlab.utils import data_dir, setup_logging  # noqa: E402

SEED = 463
N_SYNTHETIC = 1000


def lung_cancer_toy() -> pd.DataFrame:
    """ตารางสไลด์ p.40 — เรียงแถวตาม x(1), x(2), x(3)"""
    return pd.DataFrame(
        {"smoking_per_week": [3, 5, 2], "chest_pain": [1, 0, 1], "lung_cancer": [0, 1, 0]}
    )


def churn_toy() -> pd.DataFrame:
    """ตารางสไลด์ p.72"""
    return pd.DataFrame(
        {
            "monthly_usage_hours": [12, 50, 8, 30, 5],
            "subscription_type": ["Basic", "Standard", "Premium", "Standard", "Basic"],
            "churn": ["Yes", "No", "Yes", "No", "Yes"],
        }
    )


def churn_synthetic(n: int = N_SYNTHETIC, seed: int = SEED) -> pd.DataFrame:
    """ลูกค้าสมมติ n คน: ใช้งานน้อย / อยู่ไม่นาน / เปิด ticket บ่อย / แพ็กเกจ Basic → churn มากขึ้น

    ground truth คือ logistic regression จริงๆ ดังนั้น perceptron ควรเรียนสัมประสิทธิ์ใกล้เคียงนี้ได้
    """
    rng = np.random.default_rng(seed)
    sub = rng.choice(["Basic", "Standard", "Premium"], size=n, p=[0.5, 0.35, 0.15])
    usage = np.round(np.clip(rng.gamma(shape=2.0, scale=12.0, size=n), 0.5, None), 1)
    tenure = rng.integers(1, 61, size=n)
    tickets = rng.poisson(1.5, size=n)
    sub_effect = np.select([sub == "Basic", sub == "Standard"], [0.7, 0.0], default=-0.9)
    logit = 0.0 - 0.05 * usage - 0.03 * tenure + 0.45 * tickets + sub_effect
    churn = (rng.random(n) < sigmoid(logit)).astype(int)
    return pd.DataFrame(
        {
            "customer_id": np.arange(1, n + 1),
            "monthly_usage_hours": usage,
            "tenure_months": tenure,
            "support_tickets": tickets,
            "subscription_type": sub,
            "churn": churn,
        }
    )


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = StringIO()
    df.to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue().encode("utf-8")


def build_all() -> dict[str, bytes]:
    return {
        "lung_cancer_toy.csv": to_csv_bytes(lung_cancer_toy()),
        "churn_toy.csv": to_csv_bytes(churn_toy()),
        "churn_synthetic.csv": to_csv_bytes(churn_synthetic()),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="ตรวจอย่างเดียว ไม่เขียนไฟล์")
    parser.add_argument("--out", type=Path, default=None, help="โฟลเดอร์ปลายทาง (default: lab/data)")
    args = parser.parse_args()
    log = setup_logging()

    out = args.out or data_dir()
    out.mkdir(parents=True, exist_ok=True)
    files = build_all()
    ok = True
    for name, content in files.items():
        path = out / name
        if args.check:
            same = path.exists() and path.read_bytes() == content
            ok &= same
            log.info("%s %s", "ตรง" if same else "ไม่ตรง", path)
        else:
            path.write_bytes(content)
            log.info("เขียน %s (%d bytes)", path, len(content))
    if not args.check:
        df = churn_synthetic()
        log.info("churn_synthetic: churn rate = %.1f%%  (%d/%d)", 100 * df["churn"].mean(), df["churn"].sum(), len(df))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
