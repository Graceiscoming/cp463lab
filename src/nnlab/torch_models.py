"""
โมเดล PyTorch ที่คู่กับสไลด์ + TorchTrainer ที่มี interface เดียวกับโมเดล numpy
PyTorch counterparts of the lecture models plus a trainer with the shared interface

layout ของ torch: X (m, n_x) สำหรับ fully-connected และ (m, C, H, W) สำหรับภาพ (NCHW)
nn.Linear(n_in, n_out) เก็บ weight shape (n_out, n_in) = W[l] (n[l], n[l-1]) ของสไลด์พอดี
แต่คำนวณ  y = x·Wᵀ + b  เพราะ x เป็นแถว (ต่างจากสไลด์ที่ Z = W·X + b เพราะ X เป็นคอลัมน์)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .metrics import binary_report, classification_summary
from .utils import get_device


# =============================================================================
# โมเดล
# =============================================================================
class TorchPerceptron(nn.Module):
    """perceptron ตัวเดียว: คืน logit z = w·x + b (ยังไม่ผ่าน sigmoid — ใช้กับ BCEWithLogitsLoss)"""

    def __init__(self, n_in: int):
        super().__init__()
        self.linear = nn.Linear(n_in, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)          # (m, 1) → (m,)


class TorchMLP(nn.Module):
    """fully-connected network: sizes = [n_x, n[1], ..., n[L]] ; ReLU ทุก hidden ; dropout ตาม p_drop

    output = logits (1 unit → binary ใช้ BCEWithLogitsLoss ; C unit → CrossEntropyLoss)
    """

    def __init__(self, sizes: list[int], p_drop: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 2):
            layers += [nn.Linear(sizes[i], sizes[i + 1]), nn.ReLU()]
            if p_drop > 0:
                layers.append(nn.Dropout(p_drop))      # torch ใช้ p = ความน่าจะเป็นที่ "ปิด" = 1 − keep_prob
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.net = nn.Sequential(*layers)
        self.sizes = list(sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.squeeze(-1) if self.sizes[-1] == 1 else out


class LeNet5(nn.Module):
    """LeNet-5 ตามสไลด์ p.199-202: CONV(6,5×5) → POOL → CONV(16,5×5) → POOL → FC120 → FC84 → FC n_classes

    input 32×32 (pad=0) หรือ 28×28 MNIST (pad=2 ที่ conv แรก ทำให้เป็น 32×32 เสมือน) → flatten 400 เท่ากัน
    pool="max" ตาม p.199 ; "avg" ตาม LeNet-5 ดั้งเดิม p.202
    """

    def __init__(self, n_classes: int = 10, in_channels: int = 1, pad: int = 2, pool: str = "max"):
        super().__init__()
        Pool = nn.MaxPool2d if pool == "max" else nn.AvgPool2d
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=pad), nn.ReLU(), Pool(2, 2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), Pool(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    @torch.no_grad()
    def shapes(self, input_shape: tuple[int, int, int] = (1, 28, 28)) -> list[tuple[str, tuple[int, ...]]]:
        """เดิน tensor ผ่านทีละชั้นแล้วบันทึก shape — ใช้พิมพ์ตารางแบบสไลด์ p.200"""
        x = torch.zeros(1, *input_shape)
        rows = [("Input", tuple(x.shape[1:]))]
        for layer in list(self.features) + list(self.classifier):
            x = layer(x)
            if isinstance(layer, nn.ReLU):          # activation ไม่เปลี่ยน shape — ไม่ต้องแสดง
                continue
            rows.append((type(layer).__name__, tuple(x.shape[1:])))
        return rows


class ResidualBlock(nn.Module):
    """residual block แบบ fully-connected ตามสไลด์ p.205:  a[l+2] = g(z[l+2] + a[l])

    skip connection ทำให้ gradient ไหลผ่านได้ตรงๆ → เทรนเครือข่ายลึกมากได้ (p.207)
    """

    def __init__(self, n: int):
        super().__init__()
        self.fc1 = nn.Linear(n, n)
        self.fc2 = nn.Linear(n, n)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        z1 = self.fc1(a)
        a1 = torch.relu(z1)
        z2 = self.fc2(a1)
        return torch.relu(z2 + a)                   # "+ a" คือทางลัด (shortcut)


# =============================================================================
# early stopping (semantics เดียวกับ NeuralNetwork ฝั่ง numpy: stopped_epoch == best_epoch + patience)
# =============================================================================
class EarlyStopping:
    """หยุดเทรนเมื่อ validation loss ไม่ดีขึ้นติดกัน `patience` epoch แล้วคืนน้ำหนักที่ดีที่สุด (สไลด์ p.96)

    patience         จำนวน epoch ที่ยอมให้ไม่ดีขึ้น
    min_delta        ต้องดีขึ้นอย่างน้อยเท่านี้จึงนับว่า "ดีขึ้น"
    restore_best     หยุดแล้วโหลด state_dict ที่ดีที่สุดกลับเข้าโมเดล
    checkpoint_path  ถ้าระบุ จะ torch.save({"state_dict", "epoch", "val_loss"}) ทุกครั้งที่ได้ best ใหม่
    ใช้: es = EarlyStopping(5); ทุก epoch → if es.step(val_loss, model): break
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-6, restore_best: bool = True, checkpoint_path: str | Path | None = None):
        self.patience, self.min_delta, self.restore_best = patience, min_delta, restore_best
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self.best_loss = float("inf")
        self.best_epoch = -1
        self.stopped_epoch: int | None = None
        self.wait = 0
        self.epoch = -1
        self.best_state: dict | None = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """เรียกหลังจบทุก epoch — คืน True เมื่อควรหยุด"""
        self.epoch += 1
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss, self.best_epoch, self.wait = float(val_loss), self.epoch, 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if self.checkpoint_path is not None:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"state_dict": self.best_state, "epoch": self.epoch, "val_loss": self.best_loss}, self.checkpoint_path)
            return False
        self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = self.epoch
            if self.restore_best:
                self.restore(model)
            return True
        return False

    def restore(self, model: nn.Module) -> None:
        if self.best_state is not None:
            model.load_state_dict(self.best_state)

    def __repr__(self) -> str:
        return f"EarlyStopping(patience={self.patience}, best_epoch={self.best_epoch}, best_loss={self.best_loss:.4f}, stopped_epoch={self.stopped_epoch})"


# =============================================================================
# trainer ที่ทำให้ torch model ใช้ interface เดียวกับ Perceptron / NeuralNetwork
# =============================================================================
class TorchTrainer:
    """เทรน/ประเมิน nn.Module ด้วย mini-batch, optimizer ของ torch, และ history เหมือนโมเดล numpy

    loss="bce" (binary, model คืน logits (m,)) หรือ "ce" (multiclass, logits (m, C))
    optimizer = ชื่อ ("sgd" | "momentum" | "rmsprop" | "adam") หรือ torch.optim.Optimizer ที่สร้างไว้แล้ว (เช่นมี param group หลาย lr)
    param_groups = list ของ dict สำหรับ optimizer (ดู nnlab.vision.param_groups); ถ้าไม่ระบุใช้เฉพาะ parameter ที่ requires_grad
    early_stopping_patience = หยุดเมื่อ val cost ไม่ดีขึ้นติดกัน N epoch (ต้องมี val) แล้วคืนน้ำหนักที่ดีที่สุด → stopped_epoch_, best_epoch_
    รับ X เป็น numpy (m, n_x) หรือ (m, C, H, W) — แปลงเป็น float32 tensor ให้เอง
    """

    def __init__(
        self,
        model: nn.Module,
        loss: str = "bce",
        optimizer: str | torch.optim.Optimizer = "adam",
        lr: float = 0.001,
        epochs: int = 100,
        batch_size: int | None = 64,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
        device: str | None = None,
        seed: int = 463,
        verbose: bool = False,
        log_every: int = 10,
        early_stopping_patience: int | None = None,
        min_delta: float = 1e-6,
        checkpoint_path: str | Path | None = None,
        param_groups: list[dict] | None = None,
    ):
        self.model = model
        self.loss_name = loss
        self.criterion = nn.BCEWithLogitsLoss() if loss == "bce" else nn.CrossEntropyLoss()
        self.optimizer_name, self.lr, self.epochs, self.batch_size = optimizer, lr, epochs, batch_size
        self.weight_decay, self.momentum = weight_decay, momentum
        self.device = device or get_device()
        self.seed, self.verbose, self.log_every = seed, verbose, log_every
        self.early_stopping_patience, self.min_delta, self.checkpoint_path = early_stopping_patience, min_delta, checkpoint_path
        self.param_groups = param_groups
        self.stopped_epoch_: int | None = None
        self.best_epoch_: int | None = None
        self.history_: dict[str, list[float]] = {"cost": [], "val_cost": [], "val_acc": []}
        self.model.to(self.device)

    # ----- helpers ----------------------------------------------------------
    def _make_optimizer(self) -> torch.optim.Optimizer:
        if isinstance(self.optimizer_name, torch.optim.Optimizer):     # ผู้ใช้สร้างมาเอง (เช่น lr ต่างกันต่อ param group)
            return self.optimizer_name
        # ส่งเฉพาะ parameter ที่เทรนได้ — parameter ที่ freeze (requires_grad=False) ไม่ควรอยู่ใน optimizer เลย
        p = self.param_groups if self.param_groups else [q for q in self.model.parameters() if q.requires_grad]
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(p, lr=self.lr, weight_decay=self.weight_decay)
        if self.optimizer_name == "momentum":
            return torch.optim.SGD(p, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        if self.optimizer_name == "rmsprop":
            return torch.optim.RMSprop(p, lr=self.lr, weight_decay=self.weight_decay)
        return torch.optim.Adam(p, lr=self.lr, weight_decay=self.weight_decay)

    def _tensor_x(self, X) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.float().to(self.device)
        return torch.as_tensor(np.asarray(X), dtype=torch.float32, device=self.device)

    def _tensor_y(self, y) -> torch.Tensor:
        y = torch.as_tensor(np.asarray(y), device=self.device)
        return y.float() if self.loss_name == "bce" else y.long()

    def _loader(self, X, y, shuffle: bool) -> torch.utils.data.DataLoader:
        ds = torch.utils.data.TensorDataset(self._tensor_x(X), self._tensor_y(y))
        bs = self.batch_size or len(ds)
        gen = torch.Generator().manual_seed(self.seed)
        return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, generator=gen)

    # ----- การเทรน ------------------------------------------------------------
    def fit(self, X, y, X_val=None, y_val=None) -> "TorchTrainer":
        torch.manual_seed(self.seed)
        loader = self._loader(X, y, shuffle=True)
        return self.fit_loader(loader, self._loader(X_val, y_val, shuffle=False) if X_val is not None else None)

    def fit_loader(self, train_loader, val_loader=None) -> "TorchTrainer":
        opt = self._make_optimizer()
        self.history_ = {"cost": [], "val_cost": [], "val_acc": []}
        self.stopped_epoch_ = self.best_epoch_ = None
        es = EarlyStopping(self.early_stopping_patience, self.min_delta, restore_best=True, checkpoint_path=self.checkpoint_path) if (self.early_stopping_patience and val_loader is not None) else None
        for epoch in range(self.epochs):
            self.model.train()                                  # เปิด dropout / batchnorm mode เทรน
            total, n = 0.0, 0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                opt.zero_grad()                                 # ล้าง gradient เก่า (torch สะสม .grad)
                loss = self.criterion(self.model(xb), yb)
                loss.backward()                                 # backward propagation อัตโนมัติ
                opt.step()                                      # อัปเดต parameter
                total += loss.item() * len(xb)
                n += len(xb)
            self.history_["cost"].append(total / n)
            if val_loader is not None:
                val_cost, val_acc = self._evaluate_loader(val_loader)
                self.history_["val_cost"].append(val_cost)
                self.history_["val_acc"].append(val_acc)
            if self.verbose and (epoch % self.log_every == 0 or epoch == self.epochs - 1):
                msg = f"epoch {epoch:4d}  cost {self.history_['cost'][-1]:.4f}"
                if val_loader is not None:
                    msg += f"  val_cost {self.history_['val_cost'][-1]:.4f}  val_acc {self.history_['val_acc'][-1]:.4f}"
                print(msg)
            if es is not None and es.step(self.history_["val_cost"][-1], self.model):
                self.stopped_epoch_, self.best_epoch_ = es.stopped_epoch, es.best_epoch
                if self.verbose:
                    print(f"early stopping ที่ epoch {es.stopped_epoch} (best epoch {es.best_epoch}, val_cost {es.best_loss:.4f}) — คืนน้ำหนักที่ดีที่สุด")
                break
        if es is not None and self.best_epoch_ is None:
            self.best_epoch_ = es.best_epoch
        return self

    @torch.no_grad()
    def _evaluate_loader(self, loader) -> tuple[float, float]:
        self.model.eval()                                       # ปิด dropout
        total, correct, n = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            out = self.model(xb)
            total += self.criterion(out, yb).item() * len(xb)
            pred = (out >= 0).long() if self.loss_name == "bce" else out.argmax(dim=1)
            correct += (pred == yb.long()).sum().item()
            n += len(xb)
        return total / n, correct / n

    # ----- การทำนาย -----------------------------------------------------------
    @torch.no_grad()
    def predict_logits(self, X) -> np.ndarray:
        self.model.eval()
        outs = [self.model(xb.to(self.device)).cpu() for xb, in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(self._tensor_x(X)), batch_size=self.batch_size or 512)]
        return torch.cat(outs).numpy()

    def predict_proba(self, X) -> np.ndarray:
        logits = torch.as_tensor(self.predict_logits(X))
        return (torch.sigmoid(logits) if self.loss_name == "bce" else torch.softmax(logits, dim=1)).numpy()

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        P = self.predict_proba(X)
        return (P >= threshold).astype(int) if P.ndim == 1 else P.argmax(axis=1)

    def evaluate(self, X, y) -> dict[str, float]:
        y_pred = self.predict(X)
        return binary_report(y, y_pred) if self.loss_name == "bce" else classification_summary(y, y_pred)

    @torch.inference_mode()
    def predict_loader(self, loader) -> np.ndarray:
        """label ที่ทำนายของทุก batch ใน loader (ใช้กับ Dataset ของภาพที่ไม่ได้อยู่ในรูป numpy)"""
        self.model.eval()
        preds = []
        for xb, _ in loader:
            out = self.model(xb.to(self.device))
            preds.append(((out >= 0).long() if self.loss_name == "bce" else out.argmax(dim=1)).cpu())
        return torch.cat(preds).numpy()

    def evaluate_loader(self, loader) -> dict[str, float]:
        """metrics บน loader ทั้งชุด (binary_report หรือ classification_summary ตาม loss)"""
        y_true = torch.cat([torch.as_tensor(yb) for _, yb in loader]).numpy()
        y_pred = self.predict_loader(loader)
        return binary_report(y_true, y_pred) if self.loss_name == "bce" else classification_summary(y_true, y_pred)

    # ----- บันทึก/โหลด --------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        path = Path(path).with_suffix(".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict(), "history": self.history_, "loss": self.loss_name}, path)
        return path

    def load(self, path: str | Path) -> "TorchTrainer":
        """โหลด state_dict เข้า model ที่สร้างไว้แล้ว (architecture ต้องตรงกัน)"""
        ckpt = torch.load(Path(path).with_suffix(".pt"), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["state_dict"])
        self.history_ = ckpt.get("history", self.history_)
        return self

    def __repr__(self) -> str:
        opt = self.optimizer_name if isinstance(self.optimizer_name, str) else type(self.optimizer_name).__name__
        es = f", early_stopping_patience={self.early_stopping_patience}" if self.early_stopping_patience else ""
        return f"TorchTrainer({type(self.model).__name__}, loss={self.loss_name}, optimizer={opt}, lr={self.lr}, epochs={self.epochs}, batch_size={self.batch_size}, device={self.device}{es})"


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """จำนวน parameter ของโมเดล (default นับเฉพาะที่ requires_grad — หลัง freeze ตัวเลขจะลดลง)"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
