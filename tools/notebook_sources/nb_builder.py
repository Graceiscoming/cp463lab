"""
ตัวช่วยสร้าง .ipynb ด้วย nbformat — ใช้ร่วมกันทุก lab
วิธีใช้:
    from nb_builder import NB, LAB_DIR
    nb = NB()
    nb.header(5, "Perceptron จากศูนย์", "Perceptron / logistic regression from scratch", objectives=[...], slides="13-31, 40-56", minutes=120, datasets="lung_cancer_toy, make_blobs")
    nb.setup()                       # cell import + set_seed
    nb.md("## ขั้นที่ 1 ..."); nb.code("...")
    nb.convention("deck")            # banner บอก convention ที่ใช้อยู่
    nb.takeaways([...]); nb.exercises([...])
    nb.save("lab05_perceptron_from_scratch.ipynb")
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

LAB_DIR = Path(__file__).resolve().parents[2]      # lab/
NOTEBOOK_DIR = LAB_DIR / "notebooks"

GROUPS = {0: "A พื้นฐาน", 1: "A พื้นฐาน", 2: "A พื้นฐาน", 3: "A พื้นฐาน", 4: "A พื้นฐาน",
          5: "B perceptron", 6: "B perceptron", 7: "C neural network", 8: "C neural network", 9: "C neural network",
          10: "D evaluation", 11: "E CNN", 12: "E CNN", 13: "E CNN"}

CONVENTION_TEXT = {
    "deck": (
        "> **Convention ที่ใช้ในส่วนนี้: สไลด์ (deck)** — `X` มี shape `(n_x, m)` sample เป็น**คอลัมน์**, "
        "`Y` มี shape `(1, m)`, `w` มี shape `(n_x, 1)` และ `Z = w.T @ X + b`"
    ),
    "lib": (
        "> **Convention ที่ใช้ในส่วนนี้: library (pandas / scikit-learn / PyTorch)** — `X` มี shape `(m, n_x)` "
        "sample เป็น**แถว**, `y` มี shape `(m,)` และ `Z = X @ W.T + b`"
    ),
}


def _d(text: str) -> str:
    return textwrap.dedent(text).strip("\n")


def _ind(text: str) -> str:
    """indent บรรทัดต่อๆ ไปของข้อความหลายบรรทัดให้เท่า template (8 ช่อง) ไม่งั้น textwrap.dedent จะไม่ทำงาน"""
    return text.replace("\n", "\n        ")


class NB:
    def __init__(self):
        self.cells = []

    # ----- cell พื้นฐาน -----------------------------------------------------------
    def md(self, text: str) -> "NB":
        self.cells.append(new_markdown_cell(_d(text)))
        return self

    def code(self, src: str) -> "NB":
        self.cells.append(new_code_cell(_d(src)))
        return self

    # ----- cell มาตรฐานของทุก lab ---------------------------------------------------
    def header(self, n: int, title_th: str, title_en: str, objectives: list[str], slides: str, minutes: int, datasets: str = "ไม่ใช้", prereq: str | None = None) -> "NB":
        obj = _ind("\n".join(f"{i}. {o}" for i, o in enumerate(objectives, 1)))
        pre = f"\n        **ควรทำมาก่อน:** {prereq}\n" if prereq else ""
        self.md(f"""
        # Lab {n:02d} — {title_th}
        ### {title_en}

        **กลุ่ม:** {GROUPS[n]} · **อ่านคู่กับสไลด์หน้า:** {slides} · **เวลาโดยประมาณ:** {minutes} นาที · **Dataset:** {datasets}
        {pre}
        ## จุดประสงค์การเรียนรู้
        เมื่อทำ lab นี้จบ นิสิตจะสามารถ
        {obj}

        > **วิธีใช้ notebook นี้:** อ่าน markdown cell ก่อน แล้วรัน code cell ถัดไปด้วย `Shift+Enter` ทีละ cell ตามลำดับ
        > ทุก code cell มีคำอธิบายว่าทำอะไรและทำไม ถ้าอยากทดลอง ให้แก้ตัวเลขใน cell แล้วรันซ้ำได้เลย
        """)
        return self

    def setup(self, extra: str = "", note: str = "") -> "NB":
        extra = _ind(extra)   # extra หลายบรรทัดต้อง indent เท่า template ก่อน dedent
        note = _ind(note)
        self.md(f"""
        ## เตรียมเครื่องมือ
        cell แรกของทุก lab จะ import library ที่ใช้ และตั้ง **seed** ให้ตัวเลขสุ่มเหมือนกันทุกครั้งที่รัน
        (ไม่งั้นค่าเริ่มต้นของ weight จะต่างกันทุกรอบ และตัวเลขในหน้าจอจะไม่ตรงกับคำอธิบาย)
        {note}
        """)
        self.code(f"""
        import numpy as np
        import matplotlib.pyplot as plt

        try:
            import nnlab                      # package ประกอบ lab (ติดตั้งแล้วด้วย uv sync)
        except ImportError:                   # เผื่อเปิดด้วย kernel อื่น: เพิ่ม lab/src เข้า path เอง
            import sys, pathlib
            sys.path.insert(0, str(pathlib.Path.cwd().parent / "src"))
            import nnlab

        from nnlab.utils import set_seed
        {extra}
        rng = set_seed(463)                   # 463 = เลขวิชา CP463
        np.set_printoptions(precision=4, suppress=True)
        %matplotlib inline
        print("numpy", np.__version__, "| nnlab", nnlab.__version__)
        """)
        return self

    def convention(self, which: str) -> "NB":
        self.md(CONVENTION_TEXT[which])
        return self

    def takeaways(self, items: list[str]) -> "NB":
        body = _ind("\n".join(f"- {t}" for t in items))
        self.md(f"""
        ## ★ Key Takeaways
        {body}
        """)
        return self

    def exercises(self, items: list[str], hint: str = "") -> "NB":
        body = _ind("\n".join(f"{i}. {t}" for i, t in enumerate(items, 1)))
        hint = _ind(hint)
        self.md(f"""
        ## ลองทำเอง
        เพิ่ม cell ใหม่ด้านล่างแล้วลองทำ (ไม่มีเฉลยใน notebook — ใช้ผลจาก cell ก่อนหน้าตรวจคำตอบตัวเอง)

        {body}
        {hint}
        """)
        self.code("# พื้นที่สำหรับลองทำเอง\n")
        return self

    # ----- แบบฝึกหัดแบบมี guide (เพิ่ม 2026-09-13) --------------------------------------
    def exercises_intro(self, lab: int, note: str = "") -> "NB":
        """หัวข้อ "แบบฝึกหัด" + อธิบายวิธีทำและวิธีตรวจ (แทน exercises() แบบเดิม)"""
        self.md(f"""
        ## แบบฝึกหัด (ทำเอง)
        ส่วนนี้ต้อง**เขียนโค้ดเอง** — แต่ละข้อมี 3 cell: โจทย์ (อ่านให้จบก่อน) → cell โครง (มี `raise NotImplementedError` ให้แทนที่ด้วยโค้ดของคุณ) → cell ตรวจคำตอบ
        cell ตรวจคำตอบรันได้ทันที: ถ้ายังไม่ได้ทำจะขึ้น ⏳ ถ้าผิดจะขึ้น ✗ พร้อมคำใบ้ ถ้าถูกจะขึ้น ✓ PASS
        ทำได้ไม่จำกัดรอบ แก้แล้วรัน cell โครงและ cell ตรวจซ้ำ ใช้ผลจาก cell ก่อนหน้าใน lab นี้ได้ทุกตัว
        {note}
        """)
        self.code(f"""
        from nnlab.exercise import check, check_close, check_shape, summary, reset
        reset()   # ล้างผลการตรวจเก่าใน kernel นี้
        """)
        return self

    def exercise(self, ex_id: str, title: str, goal: str, steps: list[str], skeleton: str, check_code: str, hints: list[str] | None = None) -> "NB":
        """หนึ่งแบบฝึกหัด: markdown โจทย์ (+ คำใบ้พับได้) → cell โครง → cell ตรวจคำตอบ

        ex_id     เช่น "5.1" (ใช้เป็น marker ใน comment บรรทัดแรกของ cell โครง/ตรวจ ให้เครื่องมือเฉลยหาเจอ)
        skeleton  โค้ดโครงที่มี raise NotImplementedError ในจุดที่ต้องเขียน
        check_code โค้ดตรวจที่ใช้ check / check_close / check_shape (ห้าม assert)
        """
        step_text = _ind("\n".join(f"{i}. {t}" for i, t in enumerate(steps, 1)))
        hint_block = ""
        if hints:
            hint_items = _ind("\n".join(f"- {h}" for h in hints))
            hint_block = f"""
        <details><summary>คำใบ้ (คลิกเพื่อเปิด)</summary>

        {hint_items}
        </details>
        """
        self.md(f"""
        ### แบบฝึกหัด {ex_id} · {title}
        **เป้าหมาย:** {goal}

        **ขั้นตอน**
        {step_text}
        {hint_block}
        """)
        self.code(f"# --- แบบฝึกหัด {ex_id}: เขียนโค้ดของคุณใน cell นี้ (แทนที่ raise NotImplementedError) ---\n" + _d(skeleton))
        self.code(f"# --- ตรวจคำตอบ {ex_id} (รันได้ทันที ไม่ต้องแก้) ---\n" + _d(check_code))
        return self

    def exercises_summary(self) -> "NB":
        self.md("""
        ### สรุปผลแบบฝึกหัด
        รัน cell นี้หลังทำครบทุกข้อ (รัน cell ตรวจของทุกข้อก่อน)
        """)
        self.code("summary()")
        return self

    def production_note(self, module: str, what: str, step: int = 5) -> "NB":
        self.md(f"""
        ## ขั้นที่ {step} · เวอร์ชัน production ใน `nnlab`
        โค้ดที่เราเพิ่งเขียนใน cell ด้านบนคือ "โค้ดเพื่อความเข้าใจ" — แยกทุกขั้นให้เห็น
        ในการใช้งานจริง โค้ดเดียวกันถูกจัดเป็นฟังก์ชัน/คลาสใน `src/nnlab/{module}` พร้อม docstring, type hints และ test
        cell ถัดไปเรียกเวอร์ชัน production มาเทียบกับที่เราเขียนเอง ({what}) — ตัวเลขต้องตรงกัน
        เปิดไฟล์ `src/nnlab/{module}` ควบคู่กันเพื่อดูว่าโค้ด "ทดลอง" กลายเป็นโค้ด "ใช้งานจริง" ได้อย่างไร
        """)
        return self

    # ----- บันทึก ---------------------------------------------------------------------
    def save(self, filename: str) -> Path:
        nb = new_notebook(
            cells=self.cells,
            metadata={
                "kernelspec": {"name": "python3", "display_name": "Python 3 (ipykernel)", "language": "python"},
                "language_info": {"name": "python"},
            },
        )
        path = NOTEBOOK_DIR / filename
        nbf.write(nb, str(path))
        print("wrote", path, f"({len(self.cells)} cells)")
        return path
