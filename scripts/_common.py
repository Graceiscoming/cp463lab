"""ส่วนที่ script ทุกตัวใช้ร่วมกัน: เพิ่ม src/ เข้า sys.path (กรณีรันโดยไม่ได้ uv sync) และตั้ง stdout เป็น UTF-8"""
import sys
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(LAB_ROOT / "src"))

if hasattr(sys.stdout, "reconfigure"):          # Windows console ต้องตั้งเอง ไม่งั้นพิมพ์ภาษาไทยไม่ได้
    sys.stdout.reconfigure(encoding="utf-8")


def print_metrics_table(rows: list[dict], columns: list[str], label_key: str = "model") -> None:
    """พิมพ์ตาราง metrics ให้อ่านง่าย (ที่เดียวที่ script ใช้ print แทน logging)"""
    width = max(len(str(r[label_key])) for r in rows) + 2
    cw = {c: max(12, len(c) + 2) for c in columns}
    header = f"{label_key:<{width}}" + "".join(f"{c:>{cw[c]}}" for c in columns)
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c, "")
            cells.append(f"{v:>{cw[c]}.4f}" if isinstance(v, float) else f"{str(v):>{cw[c]}}")
        print(f"{str(r[label_key]):<{width}}" + "".join(cells))
