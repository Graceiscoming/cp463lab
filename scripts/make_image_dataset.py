"""
สร้าง dataset ภาพจริงขนาดเล็ก data/cifar3/ (bird, dog, frog จาก CIFAR-10) เป็นไฟล์ PNG + labels.csv แบบ deterministic
Export a 3-class CIFAR-10 subset as PNG files with a CSV registry (deterministic)

    uv run python scripts/make_image_dataset.py            # ดาวน์โหลด CIFAR-10 จาก HuggingFace (parquet ~145 MB, เร็ว) แล้ว export ~1,800 ภาพ
    uv run python scripts/make_image_dataset.py --source torchvision   # ดาวน์โหลดจาก cs.toronto.edu แทน (ช้ามาก ~50 KB/s)
    uv run python scripts/make_image_dataset.py --check    # ตรวจไฟล์ที่ commit ไว้โดยไม่ต้องดาวน์โหลด (ใช้ใน pytest)
    uv run python scripts/make_image_dataset.py --print-hash

โครงสร้างผลลัพธ์ (commit ไว้ใน repo ≈ 5 MB นิสิตไม่ต้องดาวน์โหลดอะไร)
    data/cifar3/labels.csv                path,label,split
    data/cifar3/manifest.json             จำนวนไฟล์ + sha256 ของ pixel ทุกภาพ
    data/cifar3/{train,val,test}/{bird,dog,frog}/{index:05d}.png   400 / 100 / 100 ภาพต่อ class
train และ val มาจาก CIFAR train split (ไม่ซ้ำกัน) ส่วน test มาจาก CIFAR test split (held-out ทางการ)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

import _common  # noqa: F401
from nnlab.utils import data_dir, setup_logging
from nnlab.vision import CIFAR10_IDS

SPLITS = ("train", "val", "test")
# sha256 ของ pixel ทุกภาพเรียงตาม labels.csv — ใส่ค่าหลัง export ครั้งแรก (ดู --print-hash); None = ตรวจกับ manifest.json เท่านั้น
EXPECTED_PIXEL_SHA256: str | None = "b97b28f3a15a30c0e9613347fb5690e4cc2e8e5c120026cdb4caab5e2d04e3e1"


def select_indices(labels: np.ndarray, cifar_id: int, seed: int) -> np.ndarray:
    """index ทั้งหมดของ class นี้ สับด้วย seed ที่ผูกกับ class (ไม่ขึ้นกับลำดับที่วน)"""
    idx = np.flatnonzero(labels == cifar_id)
    np.random.default_rng([seed, cifar_id]).shuffle(idx)
    return idx


def pixel_sha256(root: Path, rel_paths: list[str]) -> str:
    """hash ของ pixel (ไม่ใช่ bytes ของ PNG ซึ่งต่างกันตาม version ของ Pillow)"""
    from PIL import Image

    h = hashlib.sha256()
    for rel in rel_paths:
        with Image.open(root / rel) as im:
            h.update(np.asarray(im.convert("RGB"), dtype=np.uint8).tobytes())
    return h.hexdigest()


HF_BASE = "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text"
HF_FILES = {"train": "train-00000-of-00001.parquet", "test": "test-00000-of-00001.parquet"}


class _Source:
    """ตัวห่อให้ทั้งสองแหล่งมี interface เดียวกัน: labels (np.ndarray) และ image(i) -> np.ndarray (32, 32, 3) uint8"""

    def __init__(self, labels, get_image):
        self.labels = np.asarray(labels)
        self.image = get_image


def load_source(kind: str, split: str, log) -> _Source:
    if kind == "torchvision":                 # ต้นฉบับ cs.toronto.edu (ช้ามาก แต่เป็นทางการ)
        from torchvision.datasets import CIFAR10

        ds = CIFAR10(root=str(data_dir()), train=(split == "train"), download=True)
        return _Source(ds.targets, lambda i: ds.data[int(i)])
    # HuggingFace parquet (uoft-cs/cifar10) — ข้อมูลเดียวกัน ลำดับแถวเดียวกัน ภาพเก็บเป็น PNG bytes
    import io
    import urllib.request

    import pandas as pd
    from PIL import Image

    folder = data_dir() / "cifar10-hf"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / HF_FILES[split]
    if not path.exists():
        log.info("ดาวน์โหลด %s/%s → %s", HF_BASE, HF_FILES[split], path)
        urllib.request.urlretrieve(f"{HF_BASE}/{HF_FILES[split]}", path)
    df = pd.read_parquet(path)                # คอลัมน์: img {bytes, path}, label
    return _Source(df["label"].to_numpy(), lambda i: np.asarray(Image.open(io.BytesIO(df["img"].iloc[int(i)]["bytes"])).convert("RGB"), dtype=np.uint8))


def export(out: Path, per_class: tuple[int, int, int], seed: int, log, source: str = "hf") -> dict:
    from PIL import Image

    n_train, n_val, n_test = per_class
    log.info("โหลด CIFAR-10 จาก %s (ดาวน์โหลดครั้งแรกไปที่ %s)", source, data_dir())
    train = load_source(source, "train", log)
    test = load_source(source, "test", log)
    for split in SPLITS:                      # ล้างของเก่าเฉพาะโฟลเดอร์ split
        shutil.rmtree(out / split, ignore_errors=True)
    rows: list[tuple[str, str, str]] = []
    for cls, cid in CIFAR10_IDS.items():
        tr = select_indices(train.labels, cid, seed)
        te = select_indices(test.labels, cid, seed)
        plan = [("train", train, tr[:n_train]), ("val", train, tr[n_train : n_train + n_val]), ("test", test, te[:n_test])]
        for split, ds, sel in plan:
            folder = out / split / cls
            folder.mkdir(parents=True, exist_ok=True)
            for i in sel:
                rel = f"{split}/{cls}/{int(i):05d}.png"
                Image.fromarray(ds.image(i)).save(out / rel, format="PNG", optimize=True)
                rows.append((rel, cls, split))
    order = {s: k for k, s in enumerate(SPLITS)}
    rows.sort(key=lambda r: (order[r[2]], r[1], r[0]))
    csv_text = "path,label,split\n" + "".join(f"{p},{l},{s}\n" for p, l, s in rows)
    (out / "labels.csv").write_text(csv_text, encoding="utf-8")
    manifest = {
        "n_files": len(rows),
        "per_class": {"train": n_train, "val": n_val, "test": n_test},
        "classes": list(CIFAR10_IDS),
        "seed": seed,
        "pixel_sha256": pixel_sha256(out, [r[0] for r in rows]),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    size_mb = sum(f.stat().st_size for f in out.rglob("*.png")) / 1e6
    log.info("export %d ภาพ (%.1f MB) → %s", len(rows), size_mb, out)
    return manifest


def check(out: Path, log) -> bool:
    """ตรวจว่าไฟล์ที่มีอยู่ตรงกับ manifest (และค่าคงที่ในไฟล์นี้ถ้าตั้งไว้) โดยไม่ต้องดาวน์โหลด CIFAR"""
    csv = out / "labels.csv"
    manifest_path = out / "manifest.json"
    if not csv.exists() or not manifest_path.exists():
        log.error("ไม่พบ %s หรือ %s", csv, manifest_path)
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    lines = csv.read_text(encoding="utf-8").splitlines()
    ok = lines[0] == "path,label,split"
    rows = [line.split(",") for line in lines[1:]]
    ok &= len(rows) == manifest["n_files"]
    ok &= all(len(r) == 3 for r in rows)
    missing = [r[0] for r in rows if not (out / r[0]).exists()]
    ok &= not missing
    listed = {r[0] for r in rows}
    stray = [str(p.relative_to(out)) for p in out.rglob("*.png") if str(p.relative_to(out)) not in listed]
    ok &= not stray
    for split in SPLITS:
        for cls in manifest["classes"]:
            n = sum(1 for r in rows if r[1] == cls and r[2] == split)
            ok &= n == manifest["per_class"][split]
    digest = pixel_sha256(out, [r[0] for r in rows])
    ok &= digest == manifest["pixel_sha256"]
    if EXPECTED_PIXEL_SHA256 is not None:
        ok &= digest == EXPECTED_PIXEL_SHA256
    log.info("ไฟล์ %d | หาย %d | แปลกปลอม %d | pixel sha256 %s… %s", len(rows), len(missing), len(stray), digest[:12], "ตรง" if ok else "ไม่ตรง")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=None, help="โฟลเดอร์ปลายทาง (default: lab/data/cifar3)")
    parser.add_argument("--per-class", type=int, nargs=3, default=(400, 100, 100), metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=463)
    parser.add_argument("--source", choices=["hf", "torchvision"], default="hf", help="แหล่งดาวน์โหลด CIFAR-10 (hf = HuggingFace parquet, เร็ว)")
    parser.add_argument("--check", action="store_true", help="ตรวจอย่างเดียว ไม่ดาวน์โหลด ไม่เขียนไฟล์")
    parser.add_argument("--print-hash", action="store_true", help="พิมพ์ pixel sha256 ของไฟล์ที่มีอยู่")
    args = parser.parse_args()
    log = setup_logging()
    out = args.out or (data_dir() / "cifar3")
    if args.print_hash:
        rows = [line.split(",") for line in (out / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]]
        print(pixel_sha256(out, [r[0] for r in rows]))
        return 0
    if args.check:
        return 0 if check(out, log) else 1
    manifest = export(out, tuple(args.per_class), args.seed, log, source=args.source)
    print("pixel_sha256 =", manifest["pixel_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
