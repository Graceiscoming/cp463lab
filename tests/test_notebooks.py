"""
execute ทุก notebook ตั้งแต่ต้นจนจบ (ช้า) — รันด้วย  uv run pytest -m slow
ตั้ง env NNLAB_FAST=1 ให้ notebook ที่เทรนนาน (lab12) ใช้ข้อมูลน้อยลง
"""
import os
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").glob("lab*.ipynb"))


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[p.stem for p in NOTEBOOKS])
def test_notebook_executes(path: Path):
    os.environ["NNLAB_FAST"] = "1"
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}})
    client.execute()                      # raise CellExecutionError ถ้า cell ไหน error
    errors = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors
