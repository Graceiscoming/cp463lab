"""
เครื่องมือช่วยทั่วไป: seed, timer, logging, json, path
General helpers: seeding, timing, logging, json, paths
"""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

LOGGER_NAME = "nnlab"


def project_root() -> Path:
    """คืน path ของโฟลเดอร์ lab/ (parent ของ src/) / return the lab/ folder

    ใช้หา data/ และ runs/ ได้จากทุกที่ ไม่ว่า notebook จะถูกเปิดจาก folder ไหน
    """
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """โฟลเดอร์ data/ (override ได้ด้วย env NNLAB_DATA_DIR) / the data/ folder"""
    return Path(os.environ.get("NNLAB_DATA_DIR", project_root() / "data"))


def set_seed(seed: int = 463) -> np.random.Generator:
    """ตั้ง seed ให้ random, numpy และ torch (ถ้ามี) แล้วคืน numpy Generator
    Seed python/numpy/torch for reproducible runs; returns a numpy Generator.

    ทำไมต้องทำ: การสุ่มค่าเริ่มต้นของ weight และการสลับลำดับ mini-batch
    ทำให้ผลแต่ละครั้งไม่เท่ากัน การตั้ง seed ทำให้นิสิตทุกคนได้ตัวเลขเดียวกับใน tutorial
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # torch ไม่จำเป็นสำหรับ lab ช่วงแรก
        pass
    return np.random.default_rng(seed)


def get_device() -> str:
    """เลือก device ของ torch: cuda > mps > cpu / pick the best available torch device"""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Timer:
    """จับเวลาแบบ context manager / context-manager stopwatch

    >>> with Timer() as t:
    ...     do_work()
    >>> t.elapsed   # วินาที
    """

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc: Any) -> None:
        self.elapsed = time.perf_counter() - self.start


def setup_logging(level: str = "INFO") -> logging.Logger:
    """ตั้งค่า logging ให้ script ทุกตัวพิมพ์รูปแบบเดียวกัน / configure the nnlab logger"""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def save_json(obj: Any, path: str | Path) -> Path:
    """บันทึก dict/list เป็น JSON (แปลง numpy scalar/array ให้อัตโนมัติ)"""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
