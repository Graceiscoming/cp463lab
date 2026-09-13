import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from nb_builder import NB

nb = NB()
nb.header(0, "ตรวจ environment ก่อนเริ่ม", "Setup check",
    objectives=["ยืนยันว่า `uv sync` ติดตั้ง library ครบและ kernel ของ Jupyter ใช้ `.venv` ของ lab",
                "รู้ว่า `set_seed` ทำอะไร และทำไมทุก lab ถึงเริ่มด้วยมัน",
                "รู้จักโครงสร้างโฟลเดอร์ `lab/` และรู้ว่าจะหา source code ของ `nnlab` ได้ที่ไหน"],
    slides="–", minutes=15)
nb.md("""
## ขั้นที่ 1 · library ครบหรือไม่
ถ้า cell ถัดไปพิมพ์ version ของทุก library ได้โดยไม่มี error แสดงว่า environment พร้อมแล้ว
ถ้าขึ้น `ModuleNotFoundError` ให้กลับไปที่ terminal ในโฟลเดอร์ `lab/` แล้วรัน `uv sync` จากนั้นเลือก kernel ใหม่ (Kernel → Change Kernel)
""")
nb.code("""
import sys, platform
print("Python", sys.version.split()[0], "บน", platform.system(), platform.machine())
print("interpreter:", sys.executable)          # ควรอยู่ใน lab/.venv/

import numpy, pandas, sklearn, matplotlib, torch, torchvision
for m in (numpy, pandas, sklearn, matplotlib, torch, torchvision):
    print(f"{m.__name__:<12} {m.__version__}")
""")
nb.md("""
## ขั้นที่ 2 · package `nnlab`
`nnlab` คือ package ประกอบ lab ที่อยู่ใน `lab/src/nnlab/` — `uv sync` ติดตั้งแบบ *editable* หมายความว่าถ้าเราแก้ไฟล์ใน `src/nnlab/`
แล้ว restart kernel ก็จะเห็นผลทันทีโดยไม่ต้องติดตั้งใหม่
ทุก lab จะเขียนโค้ดเองก่อน แล้วค่อยเทียบกับเวอร์ชันใน `nnlab` ที่จัดระเบียบแล้ว (มี docstring, type hints และ test)
""")
nb.code("""
import nnlab, pathlib
print("nnlab", nnlab.__version__, "อยู่ที่", pathlib.Path(nnlab.__file__).parent)
print()
print("โมดูลที่มีให้ใช้:")
for p in sorted(pathlib.Path(nnlab.__file__).parent.glob("*.py")):
    if p.name != "__init__.py":
        first_doc = p.read_text(encoding="utf-8").split('\"\"\"')[1].strip().splitlines()[0]
        print(f"  {p.stem:<14} {first_doc}")
""")
nb.md("""
## ขั้นที่ 3 · seed คืออะไร ทำไมต้องตั้ง
คอมพิวเตอร์ "สุ่ม" ตัวเลขด้วยสูตรที่เริ่มจากค่าตั้งต้นที่เรียกว่า **seed** — seed เดียวกันให้ลำดับตัวเลขเดียวกันเสมอ
neural network เริ่มจาก weight สุ่ม ถ้าไม่ตั้ง seed ตัวเลขทุกอย่างจะเปลี่ยนทุกครั้งที่รัน ทำให้เทียบกับคำอธิบายใน notebook ไม่ได้
`set_seed(463)` ตั้ง seed ให้ทั้ง `random`, `numpy` และ `torch` แล้วคืน `numpy.random.Generator` ที่เราจะใช้สุ่มตลอด lab
""")
nb.code("""
from nnlab.utils import set_seed

rng = set_seed(463)
a = rng.normal(size=3)
rng = set_seed(463)          # ตั้ง seed เดิมอีกครั้ง
b = rng.normal(size=3)
print("รอบแรก :", a)
print("รอบสอง :", b)
print("เหมือนกันทุกตัว:", (a == b).all())
""")
nb.md("""
## ขั้นที่ 4 · ข้อมูลที่ lab ใช้
ข้อมูลเล็กอยู่ใน `lab/data/*.csv` (สร้างด้วย `scripts/make_data.py`) ส่วน dataset ของ scikit-learn อยู่ในตัว library
MNIST สำหรับ lab12 จะดาวน์โหลดอัตโนมัติครั้งแรก (~11 MB) — ถ้าอยู่ในห้องที่อินเทอร์เน็ตช้า รัน cell สุดท้ายล่วงหน้าที่บ้านได้
""")
nb.code("""
from nnlab.utils import data_dir
from nnlab.data import load_lung_cancer_toy

print("โฟลเดอร์ข้อมูล:", data_dir())
for p in sorted(data_dir().glob("*.csv")):
    print(f"  {p.name:<24} {p.stat().st_size:>7,} bytes")

X, y = load_lung_cancer_toy()
print("\\nตัวอย่าง lung_cancer_toy (สไลด์ p.40): X shape", X.shape, "y", y)
""")
nb.md("""
### (ไม่บังคับ) ดาวน์โหลดล่วงหน้าสำหรับ lab12-13
ลบ `#` หน้าบรรทัดแล้วรัน — MNIST (~11 MB) ไปที่ `lab/data/mnist/` และน้ำหนัก ResNet-18 (~45 MB) ไปที่ cache ของ torch
ทำที่บ้านครั้งเดียว ในห้องเรียนจะได้ไม่ต้องรอดาวน์โหลด
""")
nb.code("""
# from nnlab.data import load_mnist
# images, labels = load_mnist(train=True); print(images.shape, labels[:10])
# from nnlab.vision import build_resnet18_transfer
# model = build_resnet18_transfer(weights="DEFAULT"); print(model.weights_meta)
""")
nb.takeaways([
    "environment ของ lab ถูก pin version ไว้ใน `uv.lock` — ทุกคนได้ตัวเลขเดียวกัน",
    "`nnlab` เป็น package editable: แก้ไฟล์ใน `src/nnlab/` แล้ว restart kernel ก็ใช้ได้ทันที",
    "ทุก lab เริ่มด้วย `set_seed(463)` เพื่อให้ผลทำซ้ำได้",
    "ลำดับการเรียน: lab01-04 พื้นฐาน → lab05-06 perceptron → lab07-09 neural network → lab10 evaluation → lab11-12 CNN",
])
nb.exercises([
    "เปลี่ยน seed จาก 463 เป็นเลขอื่นแล้วรัน cell ขั้นที่ 3 ใหม่ — ตัวเลขเปลี่ยนไหม แล้วถ้าใช้ seed เดิมสองครั้งล่ะ",
    "เปิดไฟล์ `src/nnlab/utils.py` แล้วอ่านฟังก์ชัน `set_seed` — มัน seed อะไรบ้าง 3 อย่าง",
])
nb.save("lab00_setup_check.ipynb")
