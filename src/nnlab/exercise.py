"""
ตัวช่วยตรวจแบบฝึกหัดใน notebook: พิมพ์ PASS / FAIL / ยังไม่ได้ทำ แต่ "ไม่ raise" เพื่อให้ notebook รันต่อได้ทั้งไฟล์
Self-check helpers for notebook exercises (print-only, never raise)

รูปแบบใน notebook
    # cell โครง (skeleton) — นิสิตเขียนแทนที่ raise NotImplementedError
    def my_sigmoid(z):
        raise NotImplementedError("ยังไม่ได้ทำ")

    # cell ตรวจคำตอบ — รันได้ทันที
    from nnlab.exercise import check, check_close
    check_close("5.1 sigmoid(0) = 0.5", lambda: my_sigmoid(0.0), 0.5, hint="1 / (1 + np.exp(-z))")

สถานะที่พิมพ์: ⏳ ยังไม่ได้ทำ (เจอ NotImplementedError) · ✓ PASS · ✗ FAIL + คำใบ้ · ✗ error + ชื่อ exception
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

_RESULTS: list[tuple[str, str]] = []      # (ชื่อ, สถานะ) สะสมไว้ให้ summary()


def _report(name: str, status: str, detail: str = "", hint: str = "") -> bool:
    icon = {"pass": "✓", "fail": "✗", "todo": "⏳", "error": "✗"}[status]
    label = {"pass": "PASS", "fail": "FAIL", "todo": "ยังไม่ได้ทำ", "error": "error"}[status]
    line = f"{icon} {name}: {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    if hint and status in ("fail", "error"):
        print(f"   คำใบ้: {hint}")
    _RESULTS.append((name, status))
    return status == "pass"


def _run(fn: Callable[[], Any] | Any) -> tuple[str, Any]:
    """เรียก fn (ถ้าเป็น callable) คืน (สถานะ, ค่า)"""
    try:
        value = fn() if callable(fn) else fn
    except NotImplementedError:
        return "todo", None
    except Exception as exc:  # noqa: BLE001 — ตั้งใจดักทุกอย่างเพื่อไม่ให้ notebook หยุด
        return "error", f"{type(exc).__name__}: {exc}"
    if value is Ellipsis:                      # นิสิตปล่อย `...` ไว้ในฟังก์ชัน
        return "todo", None
    return "ok", value


def check(name: str, test: Callable[[], Any] | Any, hint: str = "") -> bool:
    """ผ่านเมื่อ test() คืนค่าที่เป็นจริง (True, array ที่ .all() จริง, ฯลฯ)"""
    status, value = _run(test)
    if status == "todo":
        return _report(name, "todo")
    if status == "error":
        return _report(name, "error", value, hint)
    try:
        ok = bool(np.all(value))
    except Exception:  # noqa: BLE001
        ok = bool(value)
    return _report(name, "pass" if ok else "fail", "", hint)


def check_close(name: str, got: Callable[[], Any] | Any, expected: Any, atol: float = 1e-6, rtol: float = 1e-5, hint: str = "") -> bool:
    """ผ่านเมื่อ got() ≈ expected (np.allclose) — พิมพ์ค่าที่ได้เทียบค่าที่คาดเมื่อไม่ผ่าน"""
    status, value = _run(got)
    if status == "todo":
        return _report(name, "todo")
    if status == "error":
        return _report(name, "error", value, hint)
    try:
        value_arr = np.asarray(value, dtype=float)
        expected_arr = np.asarray(expected, dtype=float)
        ok = value_arr.shape == expected_arr.shape and bool(np.allclose(value_arr, expected_arr, atol=atol, rtol=rtol))
    except Exception as exc:  # noqa: BLE001
        return _report(name, "error", f"เทียบค่าไม่ได้ ({type(exc).__name__}: {exc})", hint)
    detail = "" if ok else f"ได้ {np.array2string(value_arr, precision=4, suppress_small=True)} คาด {np.array2string(expected_arr, precision=4, suppress_small=True)}"
    return _report(name, "pass" if ok else "fail", detail, hint)


def check_shape(name: str, got: Callable[[], Any] | Any, shape: tuple[int, ...], hint: str = "") -> bool:
    """ผ่านเมื่อ got() มี shape ตามที่กำหนด (numpy array หรือ torch tensor)"""
    status, value = _run(got)
    if status == "todo":
        return _report(name, "todo")
    if status == "error":
        return _report(name, "error", value, hint)
    actual = tuple(getattr(value, "shape", ()))
    ok = actual == tuple(shape)
    return _report(name, "pass" if ok else "fail", "" if ok else f"ได้ shape {actual} คาด {tuple(shape)}", hint)


def summary() -> dict[str, int]:
    """สรุปจำนวน PASS / FAIL / ยังไม่ได้ทำ ของ check ทั้งหมดที่รันมาใน kernel นี้"""
    counts = {"pass": 0, "fail": 0, "todo": 0, "error": 0}
    for _, status in _RESULTS:
        counts[status] += 1
    n = len(_RESULTS)
    print(f"สรุป: PASS {counts['pass']}/{n} · FAIL {counts['fail'] + counts['error']} · ยังไม่ได้ทำ {counts['todo']}")
    if counts["pass"] == n and n > 0:
        print("🎉 ผ่านครบทุกข้อ")
    return counts


def reset() -> None:
    _RESULTS.clear()
