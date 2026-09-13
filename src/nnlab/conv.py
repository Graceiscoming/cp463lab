"""
convolution และ pooling เขียนด้วย loop ธรรมดา เพื่อให้เห็นกลไก (สไลด์ p.164-198)
Convolution and pooling with explicit loops, for understanding rather than speed

layout ของภาพ: (n_H, n_W) สำหรับ grayscale และ (n_H, n_W, n_C) สำหรับ volume — ตามสไลด์
(PyTorch ใช้ (N, C, H, W) — ดู notebook lab11 สำหรับการแปลง)

หมายเหตุ: "convolution" ในสไลด์และใน deep learning คือ cross-correlation (ไม่กลับ filter)
ซึ่งตรงกับ torch.nn.functional.conv2d แต่ไม่ตรงกับ scipy.signal.convolve2d
"""
from __future__ import annotations

import numpy as np

# ภาพ 6×6 และ filter ตรวจขอบแนวตั้งจากสไลด์ p.165 (ใช้ทำซ้ำตัวเลขใน p.166-184)
DECK_IMAGE_6x6 = np.array(
    [
        [3, 0, 1, 2, 7, 4],
        [1, 5, 8, 9, 3, 1],
        [2, 7, 2, 5, 1, 3],
        [0, 1, 3, 1, 7, 8],
        [4, 2, 1, 6, 2, 8],
        [2, 4, 5, 2, 3, 9],
    ],
    dtype=float,
)
VERTICAL_EDGE = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
HORIZONTAL_EDGE = VERTICAL_EDGE.T.copy()
# ตัวอย่าง max pooling p.193-196 (f=2, s=2 → [[9, 2], [6, 3]])
POOL_EXAMPLE_4x4 = np.array([[1, 3, 2, 1], [2, 9, 1, 1], [1, 3, 2, 3], [5, 6, 1, 2]], dtype=float)


def conv_output_size(n: int, f: int, p: int = 0, s: int = 1) -> int:
    """ขนาด output ด้านหนึ่ง = ⌊(n + 2p − f) / s⌋ + 1   (สไลด์ p.185)

    >>> conv_output_size(6, 3)            # 6×6 * 3×3 → 4×4
    4
    >>> conv_output_size(6, 3, p=1)       # same padding
    6
    >>> conv_output_size(39, 3), conv_output_size(37, 5, s=2), conv_output_size(17, 5, s=2)
    (37, 17, 7)
    """
    if n + 2 * p < f:
        raise ValueError(f"filter {f} ใหญ่กว่าภาพ {n} (+pad {p})")
    return (n + 2 * p - f) // s + 1


def zero_pad(img: np.ndarray, p: int) -> np.ndarray:
    """เติม 0 รอบขอบภาพ p pixel ทุกด้าน (สไลด์ p.183) — รองรับ (H, W) และ (H, W, C)"""
    img = np.asarray(img)
    if p == 0:
        return img
    if img.ndim == 2:
        return np.pad(img, ((p, p), (p, p)))
    if img.ndim == 3:
        return np.pad(img, ((p, p), (p, p), (0, 0)))
    raise ValueError("img ต้องเป็น 2 หรือ 3 มิติ")


def conv2d_single(img: np.ndarray, filt: np.ndarray, stride: int = 1, pad: int = 0) -> np.ndarray:
    """convolution ของภาพ grayscale (H, W) กับ filter (f, f) ด้วย loop สองชั้น (สไลด์ p.166-181)

    แต่ละช่องของ output = ผลรวมของ "คูณตำแหน่งต่อตำแหน่ง" ระหว่าง filter กับบริเวณที่ filter ทับอยู่

    >>> conv2d_single(DECK_IMAGE_6x6, VERTICAL_EDGE)
    array([[ -5.,  -4.,   0.,   8.],
           [-10.,  -2.,   2.,   3.],
           [  0.,  -2.,  -4.,  -7.],
           [ -3.,  -2.,  -3., -16.]])
    """
    img = zero_pad(np.asarray(img, dtype=float), pad)
    filt = np.asarray(filt, dtype=float)
    f = filt.shape[0]
    out_h = conv_output_size(img.shape[0], f, 0, stride)
    out_w = conv_output_size(img.shape[1], f, 0, stride)
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * stride, j * stride
            patch = img[r : r + f, c : c + f]
            out[i, j] = np.sum(patch * filt)   # Hadamard product แล้ว sum
    return out


def conv2d_volume(
    vol: np.ndarray, filters: np.ndarray, bias: np.ndarray | None = None, stride: int = 1, pad: int = 0
) -> np.ndarray:
    """convolution ของ volume (H, W, n_C) กับ filter หลายตัว (f, f, n_C, n_F) → (out_H, out_W, n_F)

    filter แต่ละตัวมีความลึกเท่า n_C ของ input (p.186) และแต่ละตัวให้ output 1 channel
    จำนวน channel ของ output = จำนวน filter n_F (p.187-188)
    """
    vol = zero_pad(np.asarray(vol, dtype=float), pad)
    filters = np.asarray(filters, dtype=float)
    if vol.ndim == 2:
        vol = vol[:, :, None]
    if filters.ndim == 3:  # filter เดียว (f, f, n_C) → (f, f, n_C, 1)
        filters = filters[:, :, :, None]
    f, _, n_c, n_f = filters.shape
    if vol.shape[2] != n_c:
        raise ValueError(f"channel ของภาพ ({vol.shape[2]}) ต้องเท่ากับความลึกของ filter ({n_c})")
    out_h = conv_output_size(vol.shape[0], f, 0, stride)
    out_w = conv_output_size(vol.shape[1], f, 0, stride)
    out = np.zeros((out_h, out_w, n_f))
    bias = np.zeros(n_f) if bias is None else np.asarray(bias, dtype=float)
    for k in range(n_f):
        for i in range(out_h):
            for j in range(out_w):
                r, c = i * stride, j * stride
                patch = vol[r : r + f, c : c + f, :]
                out[i, j, k] = np.sum(patch * filters[:, :, :, k]) + bias[k]
    return out


def max_pool2d(vol: np.ndarray, f: int = 2, s: int = 2) -> np.ndarray:
    """max pooling ทีละ channel (สไลด์ p.193-197) — ไม่มี parameter ให้เรียน

    >>> max_pool2d(POOL_EXAMPLE_4x4)
    array([[9., 2.],
           [6., 3.]])
    """
    vol = np.asarray(vol, dtype=float)
    squeeze = vol.ndim == 2
    if squeeze:
        vol = vol[:, :, None]
    out_h = conv_output_size(vol.shape[0], f, 0, s)
    out_w = conv_output_size(vol.shape[1], f, 0, s)
    out = np.zeros((out_h, out_w, vol.shape[2]))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * s, j * s
            out[i, j, :] = vol[r : r + f, c : c + f, :].max(axis=(0, 1))
    return out[:, :, 0] if squeeze else out


def avg_pool2d(vol: np.ndarray, f: int = 2, s: int = 2) -> np.ndarray:
    """average pooling (ใช้ใน LeNet-5 ดั้งเดิม p.202 — ปัจจุบันใช้น้อย)"""
    vol = np.asarray(vol, dtype=float)
    squeeze = vol.ndim == 2
    if squeeze:
        vol = vol[:, :, None]
    out_h = conv_output_size(vol.shape[0], f, 0, s)
    out_w = conv_output_size(vol.shape[1], f, 0, s)
    out = np.zeros((out_h, out_w, vol.shape[2]))
    for i in range(out_h):
        for j in range(out_w):
            r, c = i * s, j * s
            out[i, j, :] = vol[r : r + f, c : c + f, :].mean(axis=(0, 1))
    return out[:, :, 0] if squeeze else out


def describe_cnn(input_shape: tuple[int, int, int], layers: list[dict]) -> list[dict]:
    """เดิน shape ผ่านแต่ละชั้นของ CNN แล้วคืนตารางแบบสไลด์ p.200 (activation shape / size / #parameters)

    layers เป็น list ของ dict:
        {"type": "conv", "f": 5, "s": 1, "p": 0, "n_f": 6}
        {"type": "pool", "f": 2, "s": 2}
        {"type": "flatten"}
        {"type": "fc", "units": 120}

    >>> rows = describe_cnn((32, 32, 3), [{"type": "conv", "f": 5, "n_f": 6}, {"type": "pool"}])
    >>> rows[1]["shape"], rows[2]["shape"]
    ((28, 28, 6), (14, 14, 6))
    """
    h, w, c = input_shape
    rows = [{"layer": "Input", "shape": (h, w, c), "size": h * w * c, "params": 0}]
    n_conv = n_pool = n_fc = 0
    flat: int | None = None
    for layer in layers:
        t = layer["type"]
        if t == "conv":
            f, s, p, n_f = layer["f"], layer.get("s", 1), layer.get("p", 0), layer["n_f"]
            h, w = conv_output_size(h, f, p, s), conv_output_size(w, f, p, s)
            params = f * f * c * n_f + n_f          # weights + bias ต่อ filter
            c = n_f
            n_conv += 1
            rows.append({"layer": f"CONV{n_conv} (f={f}, s={s}, p={p}, n_F={n_f})", "shape": (h, w, c), "size": h * w * c, "params": params})
        elif t == "pool":
            f, s = layer.get("f", 2), layer.get("s", 2)
            h, w = conv_output_size(h, f, 0, s), conv_output_size(w, f, 0, s)
            n_pool += 1
            rows.append({"layer": f"POOL{n_pool} (f={f}, s={s})", "shape": (h, w, c), "size": h * w * c, "params": 0})
        elif t == "flatten":
            flat = h * w * c
            rows.append({"layer": "Flatten", "shape": (flat,), "size": flat, "params": 0})
        elif t == "fc":
            if flat is None:
                flat = h * w * c
                rows.append({"layer": "Flatten", "shape": (flat,), "size": flat, "params": 0})
            units = layer["units"]
            n_fc += 1
            rows.append({"layer": f"FC{n_fc}", "shape": (units,), "size": units, "params": flat * units + units})
            flat = units
        else:
            raise ValueError(f"ไม่รู้จัก layer type '{t}'")
    return rows


def format_cnn_table(rows: list[dict]) -> str:
    """แปลงผลจาก describe_cnn เป็นตารางข้อความ"""
    lines = [f"{'Layer':<32}{'Activation shape':<20}{'Size':>8}{'#Params':>10}"]
    for r in rows:
        shape = "×".join(str(d) for d in r["shape"])
        lines.append(f"{r['layer']:<32}{shape:<20}{r['size']:>8,}{r['params']:>10,}")
    lines.append(f"{'รวม parameters':<52}{sum(r['params'] for r in rows):>18,}")
    return "\n".join(lines)
