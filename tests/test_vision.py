"""tests ของ nnlab.vision — ส่วนใหญ่ offline (ภาพสุ่มใน tmp_path หรือ weights=None); ที่ต้องใช้ pretrained mark slow"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from nnlab.vision import (
    IMAGENET_MEAN, IMAGENET_STD, ImageCSVDataset, SmallCNN, TransferResNet18, build_resnet18_transfer, cifar3_root,
    denormalize, describe_trainable, extract_features, freeze_backbone, load_cifar3_loaders, make_transforms, param_groups,
    set_trainable, trainable_parameters, unfreeze_layer4,
)

LAB = Path(__file__).resolve().parents[1]
CIFAR3_READY = (cifar3_root() / "labels.csv").exists()
RESNET_CACHE = Path(torch.hub.get_dir()) / "checkpoints" / "resnet18-f37072fd.pth"


@pytest.fixture
def tiny_image_root(tmp_path, rng):
    """โฟลเดอร์ภาพสุ่ม 2 class × (4 train / 2 val / 2 test) + labels.csv"""
    rows = []
    for split, n in [("train", 4), ("val", 2), ("test", 2)]:
        for cls in ["cat", "ant"]:
            for k in range(n):
                rel = f"{split}/{cls}/{k:03d}.png"
                (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(tmp_path / rel)
                rows.append(f"{rel},{cls},{split}")
    (tmp_path / "labels.csv").write_text("path,label,split\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------- Dataset
def test_dataset_basics(tiny_image_root):
    ds = ImageCSVDataset(tiny_image_root / "labels.csv", split="train")
    assert len(ds) == 8
    assert ds.classes == ["ant", "cat"] and ds.class_to_idx == {"ant": 0, "cat": 1}   # เรียงตามชื่อ
    img, label = ds[0]
    assert img.dtype == torch.uint8 and tuple(img.shape) == (3, 32, 32) and label in (0, 1)
    assert ds.counts() == {"ant": 4, "cat": 4}
    assert set(ds.targets.tolist()) == {0, 1}
    assert "train" in repr(ds)


def test_dataset_split_validation(tiny_image_root):
    with pytest.raises(ValueError):
        ImageCSVDataset(tiny_image_root / "labels.csv", split="dev")
    (tiny_image_root / "bad.csv").write_text("file,label\na.png,x\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ImageCSVDataset(tiny_image_root / "bad.csv")
    assert len(ImageCSVDataset(tiny_image_root / "labels.csv")) == 16          # split=None → ทุกแถว


def test_shared_class_to_idx(tiny_image_root):
    train = ImageCSVDataset(tiny_image_root / "labels.csv", split="train")
    val = ImageCSVDataset(tiny_image_root / "labels.csv", split="val", class_to_idx=train.class_to_idx)
    assert val.class_to_idx == train.class_to_idx and len(val) == 4


# ---------------------------------------------------------------- transforms
def test_train_transform_random_eval_deterministic(rng):
    x = torch.from_numpy(rng.integers(0, 256, (3, 32, 32), dtype=np.uint8))
    train_tf, eval_tf = make_transforms(train=True, size=32), make_transforms(train=False, size=32)
    assert not torch.equal(train_tf(x), train_tf(x))
    assert torch.equal(eval_tf(x), eval_tf(x))
    out = eval_tf(x)
    assert out.dtype == torch.float32 and tuple(out.shape) == (3, 32, 32)
    expected = (x.float() / 255 - torch.tensor(IMAGENET_MEAN).view(3, 1, 1)) / torch.tensor(IMAGENET_STD).view(3, 1, 1)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.allclose(denormalize(out), x.float() / 255, atol=1e-6)


def test_transform_resizes(rng):
    x = torch.from_numpy(rng.integers(0, 256, (3, 32, 32), dtype=np.uint8))
    assert tuple(make_transforms(train=False, size=64)(x).shape) == (3, 64, 64)
    assert tuple(make_transforms(train=True, size=64)(x).shape) == (3, 64, 64)
    y = make_transforms(train=False, size=32, normalize=False)(x)
    assert float(y.min()) >= 0 and float(y.max()) <= 1


def test_dataloader_batches(tiny_image_root):
    ds = ImageCSVDataset(tiny_image_root / "labels.csv", split="train", transform=make_transforms(train=True, size=32))
    loader = torch.utils.data.DataLoader(ds, batch_size=3, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(1))
    xb, yb = next(iter(loader))
    assert tuple(xb.shape) == (3, 3, 32, 32) and xb.dtype == torch.float32
    assert yb.dtype == torch.int64 and tuple(yb.shape) == (3,)


# ---------------------------------------------------------------- data/cifar3 ที่ commit ไว้
@pytest.mark.skipif(not CIFAR3_READY, reason="ยังไม่มี data/cifar3 (รัน scripts/make_image_dataset.py)")
def test_cifar3_check_script_passes():
    result = subprocess.run([sys.executable, str(LAB / "scripts" / "make_image_dataset.py"), "--check"], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(not CIFAR3_READY, reason="ยังไม่มี data/cifar3")
def test_cifar3_loaders_counts():
    train, val, test, classes = load_cifar3_loaders(batch_size=64, size=32, num_workers=0)
    assert classes == ["bird", "dog", "frog"]
    assert len(train.dataset) == 1200 and len(val.dataset) == 300 and len(test.dataset) == 300
    xb, yb = next(iter(train))
    assert tuple(xb.shape) == (64, 3, 32, 32) and set(yb.tolist()) <= {0, 1, 2}
    fast_train, *_ = load_cifar3_loaders(fast=True, size=32)
    assert len(fast_train.dataset) == 300


# ---------------------------------------------------------------- โมเดล
def test_small_cnn_shapes():
    net = SmallCNN(n_classes=3)
    assert net(torch.zeros(2, 3, 32, 32)).shape == (2, 3)
    assert net(torch.zeros(2, 3, 64, 64)).shape == (2, 3)       # AdaptiveAvgPool → ไม่ผูกกับขนาดภาพ
    assert sum(p.numel() for p in net.parameters()) == 26_883


def test_transfer_resnet_freeze_counts():
    model = build_resnet18_transfer(n_classes=3, weights=None)
    t = trainable_parameters(model)
    assert t["total"] == 11_178_051 and t["trainable"] == 1_539            # 512×3 + 3
    assert unfreeze_layer4(model) == 8_395_267
    assert trainable_parameters(model)["trainable"] == 8_395_267
    assert freeze_backbone(model) == 1_539
    assert set_trainable(model, ("backbone.layer3", "backbone.layer4", "fc")) > 8_395_267


def test_transfer_resnet_bn_stays_eval_when_frozen():
    model = build_resnet18_transfer(weights=None)
    model.train()
    assert model.training and model.fc.training
    assert not model.backbone.bn1.training and not model.backbone.layer4[0].bn1.training
    unfreeze_layer4(model)
    model.train()
    assert model.backbone.layer4[0].bn1.training and not model.backbone.layer3[0].bn1.training
    running_before = model.backbone.bn1.running_mean.clone()
    model(torch.randn(4, 3, 32, 32))
    assert torch.equal(running_before, model.backbone.bn1.running_mean)      # BN ที่ freeze ไม่ขยับ


def test_transfer_resnet_forward_and_helpers():
    model = build_resnet18_transfer(n_classes=3, weights=None)
    assert model(torch.zeros(2, 3, 32, 32)).shape == (2, 3)
    assert model(torch.zeros(2, 3, 64, 64)).shape == (2, 3)
    assert model.features(torch.zeros(2, 3, 64, 64)).shape == (2, 512)
    names = [name for name, _, _ in describe_trainable(model)]
    assert names[:2] == ["backbone.conv1", "backbone.bn1"] and names[-1] == "fc"
    unfreeze_layer4(model)
    groups = param_groups(model, lr_backbone=1e-4, lr_head=1e-3)
    assert [g["lr"] for g in groups] == [1e-4, 1e-3]
    assert sum(p.numel() for p in groups[1]["params"]) == 1_539
    ds = torch.utils.data.TensorDataset(torch.randn(5, 3, 32, 32), torch.arange(5) % 3)
    F, y = extract_features(model, torch.utils.data.DataLoader(ds, batch_size=2))
    assert F.shape == (5, 512) and y.tolist() == [0, 1, 2, 0, 1]


@pytest.mark.slow
@pytest.mark.skipif(not (CIFAR3_READY and RESNET_CACHE.exists()), reason="ต้องมี data/cifar3 และน้ำหนัก ResNet-18 ใน cache")
def test_pretrained_linear_probe_beats_chance():
    from nnlab.torch_models import TorchTrainer

    train, val, _, _ = load_cifar3_loaders(fast=True, size=64, augment=False)
    model = build_resnet18_transfer(weights="DEFAULT")
    F_tr, y_tr = extract_features(model, train)
    F_va, y_va = extract_features(model, val)
    head = TorchTrainer(torch.nn.Linear(512, 3), loss="ce", lr=1e-3, epochs=30, batch_size=64, device="cpu").fit(F_tr.numpy(), y_tr.numpy())
    assert head.evaluate(F_va.numpy(), y_va.numpy())["accuracy"] >= 0.75
