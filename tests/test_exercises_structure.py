"""ตรวจโครงสร้างแบบฝึกหัดในทุก notebook (ไม่ execute): lab01-13 ต้องมีแบบฝึกหัด ≥ 2 ข้อ และทุก cell โครงต้องมี cell ตรวจคู่กัน"""
import re
from pathlib import Path

import nbformat
import pytest

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("lab*.ipynb"))
SKELETON_RE = re.compile(r"^# --- แบบฝึกหัด ([\w.]+):")
CHECK_RE = re.compile(r"^# --- ตรวจคำตอบ ([\w.]+)")


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if not p.name.startswith("lab00")], ids=[p.stem for p in NOTEBOOKS if not p.name.startswith("lab00")])
def test_exercise_cells_paired(path: Path):
    nb = nbformat.read(path, as_version=4)
    skeletons, checks = [], []
    for cell in nb.cells:
        if cell.cell_type != "code" or not cell.source:
            continue
        first = cell.source.splitlines()[0]
        if m := SKELETON_RE.match(first):
            skeletons.append(m.group(1))
            assert "NotImplementedError" in cell.source or "= ..." in cell.source, f"{path.name} แบบฝึกหัด {m.group(1)} ไม่มีจุดให้เติม"
        elif m := CHECK_RE.match(first):
            checks.append(m.group(1))
            assert "assert " not in cell.source and "raise " not in cell.source, f"{path.name} cell ตรวจ {m.group(1)} ห้าม assert/raise"
    assert len(skeletons) >= 2, f"{path.name} มีแบบฝึกหัดแค่ {len(skeletons)} ข้อ"
    assert skeletons == checks, f"{path.name} cell โครง {skeletons} กับ cell ตรวจ {checks} ไม่ตรงกัน"
    heads = [c.source.splitlines()[0] for c in nb.cells if c.cell_type == "markdown" and c.source]
    assert any(h.startswith("## แบบฝึกหัด") for h in heads), f"{path.name} ไม่มีหัวข้อแบบฝึกหัด"
