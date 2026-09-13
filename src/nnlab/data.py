"""
โหลด dataset ทุกตัวที่ใช้ใน lab และแบ่งข้อมูลแบบ stratified (สไลด์ p.122-125)
Dataset loaders and stratified splitting

ทุก loader คืน X แบบแถว = sample (m, n_x) และ y shape (m,) — library convention
(ถ้าต้องการ convention สไลด์ ใช้ nnlab.conventions.to_deck)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .utils import data_dir

# ---------------------------------------------------------------------------
# ข้อมูลเล็กจากสไลด์ (อยู่ใน data/*.csv สร้างด้วย scripts/make_data.py)
# ---------------------------------------------------------------------------
LUNG_COLUMNS = ["smoking_per_week", "chest_pain", "lung_cancer"]
CHURN_TOY_COLUMNS = ["monthly_usage_hours", "subscription_type", "churn"]


def _csv(name: str) -> Path:
    p = data_dir() / name
    if not p.exists():
        raise FileNotFoundError(f"ไม่พบ {p} — รัน  python scripts/make_data.py  ก่อน")
    return p


def load_lung_cancer_toy(as_frame: bool = False):
    """ตารางสไลด์ p.40: x1 = smoking/week, x2 = chest pain, y = lung cancer (3 แถว)

    คืน (X (3, 2), y (3,)) หรือ DataFrame ถ้า as_frame=True
    """
    import pandas as pd

    df = pd.read_csv(_csv("lung_cancer_toy.csv"))
    if as_frame:
        return df
    return df[LUNG_COLUMNS[:2]].to_numpy(dtype=float), df[LUNG_COLUMNS[2]].to_numpy(dtype=int)


def load_churn_toy():
    """ตารางสไลด์ p.72: 5 ลูกค้า (monthly usage, subscription type, churn) → DataFrame"""
    import pandas as pd

    return pd.read_csv(_csv("churn_toy.csv"))


def load_churn_synthetic(as_frame: bool = False):
    """churn dataset สังเคราะห์ ~1,000 แถว (สร้างด้วย seed ใน scripts/make_data.py)

    as_frame=True  → DataFrame ดิบ (มีคอลัมน์ subscription_type เป็นข้อความ)
    as_frame=False → (X (m, n_x) หลัง one-hot, y (m,), feature_names)
    """
    import pandas as pd

    from .preprocessing import one_hot_dataframe

    df = pd.read_csv(_csv("churn_synthetic.csv"))
    if as_frame:
        return df
    features = one_hot_dataframe(df.drop(columns=["churn", "customer_id"]))
    return features.to_numpy(dtype=float), df["churn"].to_numpy(dtype=int), list(features.columns)


# ---------------------------------------------------------------------------
# dataset ในตัว scikit-learn (offline ไม่ต้องดาวน์โหลด)
# ---------------------------------------------------------------------------
def load_breast_cancer_scaled(test_ratio: float = 0.2, seed: int = 463):
    """breast cancer (569 × 30, binary) แบ่ง train/test แบบ stratified และ normalize ด้วย μ,σ ของ train

    คืน X_train, X_test, y_train, y_test (y: 1 = malignant ตามที่กำหนดใหม่ให้ class ที่สนใจเป็น positive)
    """
    from sklearn.datasets import load_breast_cancer

    from .preprocessing import StandardNormalizer

    ds = load_breast_cancer()
    X, y = ds.data.astype(float), (ds.target == 0).astype(int)  # sklearn: 0 = malignant → ให้เป็น positive
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, test_ratio, seed)
    norm = StandardNormalizer().fit(X_tr)
    return norm.transform(X_tr), norm.transform(X_te), y_tr, y_te


def load_moons(n: int = 400, noise: float = 0.2, seed: int = 463):
    """ข้อมูล 2 มิติรูปพระจันทร์เสี้ยว 2 อัน — perceptron เส้นตรงแบ่งไม่ได้ ต้องใช้ neural network"""
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n, noise=noise, random_state=seed)
    return X.astype(float), y.astype(int)


def load_blobs(n: int = 200, seed: int = 463, cluster_std: float = 1.2):
    """ข้อมูล 2 มิติ 2 กลุ่มที่แบ่งด้วยเส้นตรงได้ — สำหรับวาด decision boundary ของ perceptron"""
    from sklearn.datasets import make_blobs

    X, y = make_blobs(n_samples=n, centers=2, cluster_std=cluster_std, random_state=seed)
    return X.astype(float), y.astype(int)


def load_digits_images(classes: list[int] | None = None):
    """ภาพตัวเลข 8×8 ของ scikit-learn (1,797 ภาพ) คืน (images (m, 8, 8) float 0-16, y (m,))"""
    from sklearn.datasets import load_digits

    ds = load_digits()
    images, y = ds.images.astype(float), ds.target.astype(int)
    if classes is not None:
        mask = np.isin(y, classes)
        images, y = images[mask], y[mask]
        remap = {c: i for i, c in enumerate(classes)}
        y = np.vectorize(remap.get)(y)
    return images, y


# ---------------------------------------------------------------------------
# MNIST ผ่าน torchvision (ดาวน์โหลดครั้งแรก ~11MB ไว้ที่ data/mnist/)
# ---------------------------------------------------------------------------
def load_mnist(train: bool = True, limit: int | None = None, root: str | Path | None = None, download: bool = True):
    """คืน (images (m, 28, 28) uint8, labels (m,)) จาก torchvision.datasets.MNIST

    limit = จำนวนภาพที่ต้องการ (สุ่มแบบ stratified ด้วย seed 463) ใช้ลดเวลาในห้องเรียน
    """
    from torchvision.datasets import MNIST

    root = Path(root) if root is not None else data_dir() / "mnist"
    ds = MNIST(root=str(root), train=train, download=download)
    images = ds.data.numpy()
    labels = ds.targets.numpy()
    if limit is not None and limit < len(labels):
        keep, _, _, _ = stratified_split(np.arange(len(labels)), labels, test_ratio=1 - limit / len(labels), seed=463)
        images, labels = images[keep], labels[keep]
    return images, labels


# ---------------------------------------------------------------------------
# การแบ่งข้อมูล (สไลด์ p.122-125)
# ---------------------------------------------------------------------------
def stratified_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 463):
    """แบ่ง train/test โดยรักษาสัดส่วนของแต่ละ class ให้เท่าเดิม (stratified sampling)

    ทำเองด้วย numpy เพื่อให้เห็นกลไก: สุ่มภายในแต่ละ class แยกกัน แล้วตัดตามสัดส่วน
    คืน X_train, X_test, y_train, y_test
    """
    X = np.asarray(X)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_ratio))
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])
    train_idx = rng.permutation(np.array(train_idx, dtype=int))
    test_idx = rng.permutation(np.array(test_idx, dtype=int))
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def kfold_indices(y: np.ndarray, k: int = 5, seed: int = 463):
    """สร้าง index สำหรับ k-fold cross validation แบบ stratified (สไลด์ p.124-125)

    คืน list ของ (train_idx, val_idx) จำนวน k คู่ — ทุก sample เป็น validation ครั้งเดียว
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    folds: list[list[int]] = [[] for _ in range(k)]
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        for i, chunk in enumerate(np.array_split(idx, k)):
            folds[i].extend(chunk.tolist())
    out = []
    all_idx = np.arange(len(y))
    for i in range(k):
        val = np.array(sorted(folds[i]), dtype=int)
        train = np.setdiff1d(all_idx, val)
        out.append((train, val))
    return out


DATASETS = ["lung", "churn", "breast_cancer", "moons", "blobs", "digits"]


def get_dataset(name: str, seed: int = 463):
    """ตัวเลือก dataset สำหรับ scripts/train.py → คืน X_train, X_test, y_train, y_test, n_classes"""
    from .preprocessing import StandardNormalizer

    if name == "lung":
        X, y = load_lung_cancer_toy()
        return X, X, y, y, 2  # เล็กเกินจะแบ่ง — ใช้ชุดเดียวกันเพื่อสาธิต
    if name == "breast_cancer":
        X_tr, X_te, y_tr, y_te = load_breast_cancer_scaled(seed=seed)
        return X_tr, X_te, y_tr, y_te, 2
    if name == "churn":
        X, y, _ = load_churn_synthetic()
    elif name == "moons":
        X, y = load_moons(seed=seed)
    elif name == "blobs":
        X, y = load_blobs(seed=seed)
    elif name == "digits":
        images, y = load_digits_images()
        X = images.reshape(len(images), -1)
    else:
        raise ValueError(f"ไม่รู้จัก dataset '{name}' — เลือกจาก {DATASETS}")
    X_tr, X_te, y_tr, y_te = stratified_split(X, y, 0.2, seed)
    norm = StandardNormalizer().fit(X_tr)
    return norm.transform(X_tr), norm.transform(X_te), y_tr, y_te, int(len(np.unique(y)))
