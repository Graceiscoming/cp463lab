"""
ตรวจว่าแบบฝึกหัดใน notebook "ทำได้จริง": แทนที่ cell โครงด้วยเฉลย แล้ว execute ทั้ง notebook ดูว่า cell ตรวจพิมพ์ PASS ครบ
Verify notebook exercises are solvable: inject solutions, execute, and require every check to PASS

    uv run python tools/exercise_check.py notebooks/lab05_perceptron_from_scratch.ipynb ../lab-solutions/lab05_solutions.py
    uv run python tools/exercise_check.py notebooks/lab05_*.ipynb            # ไม่มีเฉลย → รายงานสถานะ ⏳ ของทุกข้อ (ต้องไม่มี error)

ไฟล์เฉลยเป็น Python ที่มี dict SOLUTIONS = {"5.1": "โค้ดเต็มของ cell โครง", ...}
cell โครงถูกระบุด้วยบรรทัดแรก  "# --- แบบฝึกหัด <id>:"  และ cell ตรวจด้วย  "# --- ตรวจคำตอบ <id>"
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

SKELETON_RE = re.compile(r"^# --- แบบฝึกหัด ([\w.]+):")
CHECK_RE = re.compile(r"^# --- ตรวจคำตอบ ([\w.]+)")


def load_solutions(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    spec = importlib.util.spec_from_file_location("solutions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(mod.SOLUTIONS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", type=Path)
    parser.add_argument("solutions", type=Path, nargs="?", default=None)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    solutions = load_solutions(args.solutions)
    nb = nbformat.read(args.notebook, as_version=4)
    injected, ids = 0, []
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        first = cell.source.splitlines()[0] if cell.source else ""
        m = SKELETON_RE.match(first)
        if m:
            ids.append(m.group(1))
            if m.group(1) in solutions:
                cell.source = first + "\n" + solutions[m.group(1)].strip("\n")
                injected += 1
    missing = [i for i in ids if i not in solutions]
    print(f"{args.notebook.name}: แบบฝึกหัด {len(ids)} ข้อ {ids} | ใส่เฉลย {injected} ข้อ" + (f" | ไม่มีเฉลย {missing}" if missing and solutions else ""))

    os.environ.setdefault("NNLAB_FAST", "1")
    client = NotebookClient(nb, timeout=args.timeout, kernel_name="python3", resources={"metadata": {"path": str(args.notebook.parent)}}, allow_errors=True)
    client.execute()

    errors = [(i, o) for i, c in enumerate(nb.cells) if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    status_lines = []
    for cell in nb.cells:
        if cell.cell_type == "code" and CHECK_RE.match(cell.source.splitlines()[0] if cell.source else ""):
            text = "".join(o.get("text", "") for o in cell.get("outputs", []) if o.get("output_type") == "stream")
            status_lines += [line for line in text.splitlines() if line.startswith(("✓", "✗", "⏳"))]
    for line in status_lines:
        print("  ", line)
    n_pass = sum(line.startswith("✓") for line in status_lines)
    n_fail = sum(line.startswith("✗") for line in status_lines)
    n_todo = sum(line.startswith("⏳") for line in status_lines)
    print(f"PASS {n_pass} · FAIL {n_fail} · ยังไม่ได้ทำ {n_todo} · cell error {len(errors)}")
    for i, o in errors:
        print(f"  cell {i}: {o.get('ename')}: {str(o.get('evalue'))[:200]}")
    if errors:
        return 2
    if solutions and (n_fail or n_todo or missing):
        return 1
    if not solutions and n_fail:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
