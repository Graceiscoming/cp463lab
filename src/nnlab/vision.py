"""
เครื่องมือสำหรับงาน computer vision จริง: Dataset จากไฟล์ภาพ, transform, CNN เล็ก และ transfer learning ด้วย ResNet-18
Practical vision toolkit: image-file Dataset, train/eval transforms, a small CNN, ResNet-18 transfer learning helpers

ใช้ใน lab13 และ scripts/train_transfer.py
layout ของภาพเป็นแบบ torch เสมอ: uint8 (C, H, W) จาก decode_image → float32 ช่วง [0, 1] → normalize ด้วยค่าเฉลี่ย/ส่วนเบี่ยงเบนของ ImageNet
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.io import ImageReadMode, decode_image
from torchvision.transforms import v2

from .utils import data_dir, project_root

# ค่าเฉลี่ย/ส่วนเบี่ยงเบนของ ImageNet ที่ pretrained model ของ torchvision ใช้ตอนเทรน — ต้องใช้ชุดเดียวกันตอน transfer learning
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR3_CLASSES = ("bird", "dog", "frog")
CIFAR10_IDS = {"bird": 2, "dog": 5, "frog": 6}          # index ของ class ใน CIFAR-10 ต้นทาง


# =============================================================================
# Dataset จากไฟล์ภาพ + CSV registry
# =============================================================================
def read_image_rgb(path: str | Path) -> torch.Tensor:
    """อ่านไฟล์ภาพเป็น uint8 tensor (3, H, W) — อ่าน bytes เองแล้วค่อย decode เพื่อรองรับ path ที่มีภาษาไทยบน Windows"""
    data = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.uint8)
    return decode_image(data, mode=ImageReadMode.RGB)


class ImageCSVDataset(torch.utils.data.Dataset):
    """Dataset ที่อ่านภาพจากโฟลเดอร์ตามรายการใน CSV (คอลัมน์ path, label และ split ถ้ามี)

    csv_path   ไฟล์ CSV; path ในไฟล์เป็น path สัมพัทธ์กับ root
    root       โฟลเดอร์ฐานของภาพ (default = โฟลเดอร์ที่ CSV อยู่)
    split      เลือกเฉพาะแถวที่ split ตรงกัน เช่น "train" (None = ทุกแถว)
    transform  callable ที่รับ uint8 (3, H, W) แล้วคืน tensor ที่โมเดลต้องการ (ดู make_transforms)
    หลังสร้าง: classes (sorted), class_to_idx, paths, targets (int64) — label เป็น class index ไม่ใช่ one-hot
    __getitem__(i) → (image, int label) โดยเรียก transform ใหม่ทุกครั้ง (augmentation จึงสุ่มต่างกันทุก epoch)
    """

    def __init__(self, csv_path: str | Path, root: str | Path | None = None, split: str | None = None, transform=None, class_to_idx: dict[str, int] | None = None):
        import pandas as pd

        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        missing = {"path", "label"} - set(df.columns)
        if missing:
            raise ValueError(f"CSV ต้องมีคอลัมน์ {sorted(missing)} (มี {list(df.columns)})")
        if split is not None:
            if "split" not in df.columns:
                raise ValueError("CSV ไม่มีคอลัมน์ split จึงเลือก split ไม่ได้")
            available = sorted(df["split"].unique())
            if split not in available:
                raise ValueError(f"ไม่รู้จัก split '{split}' — มีให้เลือก {available}")
            df = df[df["split"] == split]
        self.csv_path = csv_path
        self.root = Path(root) if root is not None else csv_path.parent
        self.split = split
        self.transform = transform
        if class_to_idx is None:
            self.classes = sorted(df["label"].unique().tolist())
            self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        else:
            self.class_to_idx = dict(class_to_idx)
            self.classes = sorted(self.class_to_idx, key=self.class_to_idx.get)
        self.labels: list[str] = df["label"].tolist()
        self.paths: list[Path] = [self.root / p for p in df["path"].tolist()]
        self.targets = np.array([self.class_to_idx[label] for label in self.labels], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.paths)

    def load_image(self, i: int) -> torch.Tensor:
        """ภาพดิบ uint8 (3, H, W) ยังไม่ผ่าน transform — ใช้ตอนอยากดูภาพจริง"""
        return read_image_rgb(self.paths[i])

    def __getitem__(self, i: int):
        img = self.load_image(i)
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.targets[i])

    def counts(self) -> dict[str, int]:
        """จำนวนภาพต่อ class"""
        return {c: int((self.targets == idx).sum()) for c, idx in self.class_to_idx.items()}

    def __repr__(self) -> str:
        return f"ImageCSVDataset(split={self.split!r}, n={len(self)}, classes={self.classes}, root={self.root})"


def make_transforms(train: bool, size: int = 64, normalize: bool = True) -> v2.Compose:
    """transform สำหรับ train (มี augmentation) หรือ eval (deterministic) — รับ uint8 (3, H, W) คืน float32 (3, size, size)

    train: Resize → RandomCrop (pad ขอบแล้วตัดสุ่ม) → RandomHorizontalFlip → ToDtype(float, scale) → Normalize
    eval : Resize → ToDtype → Normalize
    augmentation ใช้เฉพาะ train เพราะ val/test ต้องวัดจากภาพจริงที่ไม่แต่ง
    """
    ops: list = [v2.Resize(size, antialias=True)]
    if train:
        ops += [v2.RandomCrop(size, padding=max(size // 8, 1)), v2.RandomHorizontalFlip()]
    ops.append(v2.ToDtype(torch.float32, scale=True))       # uint8 0-255 → float 0-1
    if normalize:
        ops.append(v2.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    return v2.Compose(ops)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """ย้อน Normalize เพื่อเอาภาพไปแสดง (คืนค่าใน [0, 1])"""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x * std + mean).clamp(0, 1)


def cifar3_root(root: str | Path | None = None) -> Path:
    """โฟลเดอร์ data/cifar3 (เคารพ NNLAB_DATA_DIR แต่ fallback ไปที่ไฟล์ที่ commit ไว้ใน repo)"""
    if root is not None:
        return Path(root)
    candidate = data_dir() / "cifar3"
    if candidate.exists():
        return candidate
    return project_root() / "data" / "cifar3"


def load_cifar3_loaders(batch_size: int = 64, size: int = 64, augment: bool = True, fast: bool = False, root: str | Path | None = None, num_workers: int = 0, seed: int = 463):
    """สร้าง DataLoader ของ train/val/test จาก data/cifar3 → (train_loader, val_loader, test_loader, classes)

    fast=True ใช้ train แค่ 100 ภาพแรกต่อ class (สำหรับ NNLAB_FAST / ทดสอบ)
    num_workers=0 เป็นค่าที่ปลอดภัยใน notebook บน macOS/Windows
    """
    base = cifar3_root(root)
    csv = base / "labels.csv"
    if not csv.exists():
        raise FileNotFoundError(f"ไม่พบ {csv} — รัน  python scripts/make_image_dataset.py  หรือ pull ข้อมูลจาก repo")
    train_ds = ImageCSVDataset(csv, split="train", transform=make_transforms(train=augment, size=size))
    val_ds = ImageCSVDataset(csv, split="val", transform=make_transforms(train=False, size=size), class_to_idx=train_ds.class_to_idx)
    test_ds = ImageCSVDataset(csv, split="test", transform=make_transforms(train=False, size=size), class_to_idx=train_ds.class_to_idx)
    train_set: torch.utils.data.Dataset = train_ds
    if fast:
        keep = np.concatenate([np.flatnonzero(train_ds.targets == c)[:100] for c in range(len(train_ds.classes))])
        train_set = torch.utils.data.Subset(train_ds, keep.tolist())
    gen = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=gen, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, list(train_ds.classes)


# =============================================================================
# โมเดล
# =============================================================================
class SmallCNN(nn.Module):
    """CNN เล็กสำหรับเทรนจากศูนย์: 3 block (conv 3×3 → BatchNorm → ReLU → MaxPool) → AdaptiveAvgPool(4) → Dropout → Linear

    AdaptiveAvgPool ทำให้ใช้ได้กับภาพทุกขนาด (32, 64, ...) โดยจำนวน parameter เท่าเดิม (~94k เมื่อ channels=(16, 32, 64))
    output = logits (N, n_classes) สำหรับ CrossEntropyLoss
    """

    def __init__(self, n_classes: int = 3, channels: Sequence[int] = (16, 32, 64), p_drop: float = 0.25, in_channels: int = 3):
        super().__init__()
        blocks: list[nn.Module] = []
        c_in = in_channels
        for c_out in channels:
            blocks += [nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c_out
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(p_drop), nn.Linear(c_in * 16, n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


class TransferResNet18(nn.Module):
    """ResNet-18 จาก torchvision ที่แทน fc ด้วย head ใหม่ขนาด n_classes

    weights="DEFAULT" → โหลด ResNet18_Weights.DEFAULT (ImageNet; ดาวน์โหลด ~45 MB ครั้งแรกไปที่ torch cache)
    weights=None      → สุ่ม (สำหรับ test/offline)
    weights=<path>    → โหลด state_dict ของ resnet18 จากไฟล์ (ห้องเรียนที่เน็ตช้า)

    train(mode) ถูก override: BatchNorm ของ backbone ที่ถูก freeze (weight.requires_grad == False) จะอยู่ใน eval() เสมอ
    ไม่งั้น model.train() จะทำให้ running mean/var ของ BN เปลี่ยนไปตามข้อมูลใหม่ทั้งที่เราตั้งใจ "แช่แข็ง" backbone
    """

    def __init__(self, n_classes: int = 3, weights: str | Path | None = "DEFAULT"):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        self.weights_meta: dict = {}
        if weights == "DEFAULT":
            w = ResNet18_Weights.DEFAULT
            backbone = resnet18(weights=w)
            self.weights_meta = {"name": str(w), "acc@1": w.meta.get("_metrics", {}).get("ImageNet-1K", {}).get("acc@1"), "n_categories": len(w.meta.get("categories", []))}
        elif weights is None:
            backbone = resnet18(weights=None)
        else:
            backbone = resnet18(weights=None)
            state = torch.load(Path(weights), map_location="cpu", weights_only=True)
            backbone.load_state_dict(state)
        backbone.fc = nn.Identity()                  # ตัด head 1000 class ของ ImageNet ออก → output คือ feature 512 มิติ
        self.backbone = backbone
        self.fc = nn.Linear(512, n_classes)          # head ใหม่สำหรับโจทย์ของเรา
        self.n_classes = n_classes

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """feature vector (N, 512) จาก backbone"""
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.backbone(x))

    def train(self, mode: bool = True) -> "TransferResNet18":
        super().train(mode)
        if mode:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d) and not m.weight.requires_grad:
                    m.eval()                         # BN ที่ freeze แล้ว ห้ามอัปเดต running stats
        return self


def set_trainable(model: nn.Module, patterns: Sequence[str]) -> int:
    """ตั้ง requires_grad=True เฉพาะ parameter ที่ชื่อขึ้นต้นด้วย pattern ใดๆ (ที่เหลือ False) คืนจำนวน parameter ที่เทรนได้"""
    n = 0
    for name, p in model.named_parameters():
        p.requires_grad = any(name.startswith(pat) for pat in patterns)
        n += p.numel() if p.requires_grad else 0
    return n


def freeze_backbone(model: TransferResNet18) -> int:
    """แช่แข็งทุกอย่างยกเว้น head (fc) — ขั้นแรกของ transfer learning"""
    return set_trainable(model, ("fc",))


def unfreeze_layer4(model: TransferResNet18) -> int:
    """ปลด freeze block สุดท้าย (layer4) + fc สำหรับ fine-tune — ชั้นท้ายเรียน feature เฉพาะโจทย์ ชั้นต้นเก็บ feature ทั่วไปไว้"""
    return set_trainable(model, ("backbone.layer4", "fc"))


def build_resnet18_transfer(n_classes: int = 3, freeze: bool = True, weights: str | Path | None = "DEFAULT") -> TransferResNet18:
    """สร้าง ResNet-18 สำหรับ transfer learning: โหลดน้ำหนัก → แทน fc → (freeze backbone)"""
    model = TransferResNet18(n_classes=n_classes, weights=weights)
    if freeze:
        freeze_backbone(model)
    return model


def trainable_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total, "frozen": total - trainable}


def describe_trainable(model: nn.Module) -> list[tuple[str, int, bool]]:
    """รายการ (ชื่อ module, จำนวน parameter, เทรนได้ไหม) ต่อ module ระดับบน (ลงไปหนึ่งชั้นใน backbone)"""
    rows: list[tuple[str, int, bool]] = []

    def add(prefix: str, module: nn.Module) -> None:
        params = list(module.parameters())
        if not params:
            return
        rows.append((prefix, sum(p.numel() for p in params), any(p.requires_grad for p in params)))

    for name, child in model.named_children():
        if name == "backbone":
            for sub_name, sub in child.named_children():
                add(f"backbone.{sub_name}", sub)
        else:
            add(name, child)
    return rows


def format_trainable_table(model: nn.Module) -> str:
    lines = [f"{'module':<20}{'parameters':>14}  trainable"]
    for name, n, trainable in describe_trainable(model):
        lines.append(f"{name:<20}{n:>14,}  {'yes' if trainable else 'no'}")
    t = trainable_parameters(model)
    lines.append(f"{'รวม':<20}{t['total']:>14,}  trainable {t['trainable']:,} ({100 * t['trainable'] / max(t['total'], 1):.2f}%)")
    return "\n".join(lines)


def param_groups(model: nn.Module, lr_backbone: float, lr_head: float, head_prefix: str = "fc") -> list[dict]:
    """แบ่ง parameter ที่เทรนได้เป็น 2 กลุ่มให้ optimizer ใช้ learning rate ต่างกัน (backbone ต่ำ, head สูง)"""
    head = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(head_prefix)]
    body = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith(head_prefix)]
    groups = []
    if body:
        groups.append({"params": body, "lr": lr_backbone})
    if head:
        groups.append({"params": head, "lr": lr_head})
    return groups


@torch.inference_mode()
def extract_features(model: TransferResNet18, loader: torch.utils.data.DataLoader, device: str = "cpu"):
    """ผ่านทุกภาพใน loader เข้า backbone ครั้งเดียว → (features (N, 512), labels (N,)) — backbone แช่แข็ง = feature extractor"""
    model = model.to(device).eval()
    feats, ys = [], []
    for xb, yb in loader:
        feats.append(model.features(xb.to(device)).cpu())
        ys.append(torch.as_tensor(yb))
    return torch.cat(feats), torch.cat(ys)
