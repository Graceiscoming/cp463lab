# CP463 Lab — Perceptron, Neural Network และ Convolutional Neural Network

ชุด tutorial ปฏิบัติที่คู่กับสไลด์บรรยาย *Perceptron, Neural Network, and Convolutional Neural Network* (210 หน้า)
สำหรับนิสิตที่ยังไม่มีพื้นฐาน เริ่มจาก data structure ที่ต้องใช้ (numpy → pandas → torch tensor)
ไปจนถึงการสร้างและประเมินโมเดลด้วยเครื่องมือ 3 ระดับ

| ระดับ | เครื่องมือ | เห็นอะไร |
|---|---|---|
| จากศูนย์ | numpy, pandas | ทุกสูตรในสไลด์กลายเป็นโค้ดทีละบรรทัด |
| สำเร็จรูป | scikit-learn | โมเดลเดียวกันใน 3 บรรทัด และ metric มาตรฐาน |
| ระดับใช้งานจริง | PyTorch | tensor, autograd, `nn.Module`, `DataLoader`, CNN |

แต่ละหัวข้อมี 2 รูปแบบ
- **notebook** (`notebooks/labNN_*.ipynb`) เพื่อความเข้าใจ — ทุก code cell มี markdown อธิบาย และรันเก็บ output ไว้แล้ว อ่านได้เลยโดยไม่ต้องรัน
- **package + scripts** (`src/nnlab/`, `scripts/`) เพื่อดูว่าโค้ดเดียวกันเขียนแบบ production อย่างไร (ฟังก์ชัน/คลาส, type hints, CLI, logging, pytest)

---

## สำหรับนิสิต: ติดตั้งและรัน lab (macOS / Windows / Linux)

ต้องมีแค่ 2 อย่าง: **Git** (ดาวน์โหลด lab) และ **uv** (ติดตั้ง Python และ library ทั้งหมดให้อัตโนมัติ — ไม่ต้องลง Python เอง)
ใช้พื้นที่ดิสก์ประมาณ 2 GB (PyTorch + Jupyter) และครั้งแรกใช้เวลาดาวน์โหลด 3-10 นาทีตามความเร็วอินเทอร์เน็ต

### ขั้นที่ 1 · ติดตั้ง Git และ uv (ทำครั้งเดียว)

<details><summary><b>macOS</b></summary>

เปิด **Terminal** (Spotlight → พิมพ์ Terminal) แล้วรันทีละบรรทัด

```bash
xcode-select --install          # ติดตั้ง Git (ถ้ามีอยู่แล้วจะบอกว่า already installed ข้ามได้)
curl -LsSf https://astral.sh/uv/install.sh | sh
```
ปิดแล้วเปิด Terminal ใหม่ จากนั้นพิมพ์ `uv --version` ต้องเห็นเลข version
</details>

<details><summary><b>Windows 10 / 11</b></summary>

เปิด **PowerShell** (Start → พิมพ์ PowerShell) แล้วรันทีละบรรทัด

```powershell
winget install --id Git.Git -e --source winget
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
ปิดแล้วเปิด PowerShell ใหม่ จากนั้นพิมพ์ `uv --version` ต้องเห็นเลข version

ข้อควรระวังบน Windows
- ถ้าชื่อผู้ใช้ Windows เป็นภาษาไทยหรือมีช่องว่าง ให้เก็บ lab ไว้ที่ path สั้นๆ ที่เป็นภาษาอังกฤษ เช่น `C:\lab` (บางเครื่องมืออ่าน path ภาษาไทยไม่ได้)
- ถ้า `winget` ไม่มี ให้ดาวน์โหลด Git จาก https://git-scm.com/download/win แล้วติดตั้งแบบค่า default
</details>

<details><summary><b>Linux (Ubuntu / Debian)</b></summary>

```bash
sudo apt update && sudo apt install -y git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```
ปิดแล้วเปิด terminal ใหม่ (หรือ `source ~/.bashrc`) จากนั้น `uv --version`
บน Linux ไฟล์ตั้งค่าของ lab จะเลือก PyTorch แบบ CPU ให้อัตโนมัติ (ไม่ต้องดาวน์โหลด CUDA 2 GB)
</details>

### ขั้นที่ 2 · ดาวน์โหลด lab และติดตั้ง library (ทำครั้งเดียว)

**สำคัญ: `git clone` จะสร้างโฟลเดอร์ `neural-network-lab` ไว้ "ในโฟลเดอร์ที่ terminal อยู่ตอนนั้น"** ดังนั้นต้อง `cd` ไปยังที่ที่อยากเก็บ lab ก่อน
(terminal ที่เพิ่งเปิดจะอยู่ที่โฟลเดอร์บ้านของผู้ใช้ — ดูตำแหน่งปัจจุบันได้ด้วยคำสั่ง `pwd`)

<details><summary><b>macOS / Linux</b> — เก็บไว้ที่ Documents</summary>

```bash
cd ~/Documents
pwd                      # ต้องขึ้น /Users/<ชื่อคุณ>/Documents (macOS) หรือ /home/<ชื่อคุณ>/Documents (Linux)
```
</details>

<details><summary><b>Windows</b> — เก็บไว้ที่ C:\lab (path สั้น ไม่มีภาษาไทยหรือช่องว่าง)</summary>

```powershell
mkdir C:\lab
cd C:\lab
pwd                      # ต้องขึ้น C:\lab
```
</details>

จากนั้นรัน 3 คำสั่งนี้ (เหมือนกันทุกระบบ) — บรรทัดที่สองคือการ "เข้าไปใน" โฟลเดอร์ที่เพิ่ง clone มา

```bash
git clone https://github.com/VRU-AI-SWU/neural-network-lab.git
cd neural-network-lab
uv sync
```
หลัง `cd neural-network-lab` ลองพิมพ์ `ls` (Windows ใช้ `dir` ได้เช่นกัน) ต้องเห็น `pyproject.toml`, `notebooks`, `src` — ถ้าไม่เห็น แสดงว่ายังไม่ได้อยู่ในโฟลเดอร์ lab
`uv sync` จะดาวน์โหลด Python 3.10 (ถ้าเครื่องยังไม่มี) และติดตั้ง numpy, pandas, scikit-learn, PyTorch, Jupyter ตาม `uv.lock`
ลงในโฟลเดอร์ `.venv` ภายใน lab — ทุกคนได้ version เดียวกัน ไม่กระทบ Python อื่นในเครื่อง
โฟลเดอร์ที่ได้จะอยู่ที่ `~/Documents/neural-network-lab` (macOS/Linux) หรือ `C:\lab\neural-network-lab` (Windows) เปิดดูใน Finder / File Explorer ได้ตามปกติ

### ขั้นที่ 3 · เปิด notebook (ทำทุกครั้งที่จะเรียน)

คำสั่ง `uv run ...` ต้องรัน**จากในโฟลเดอร์ `neural-network-lab`** — ทุกครั้งที่เปิด terminal ใหม่ให้ `cd` เข้าไปก่อน

```bash
cd ~/Documents/neural-network-lab        # Windows: cd C:\lab\neural-network-lab
uv run jupyter lab
```
browser จะเปิดขึ้นเอง (ถ้าไม่เปิด ให้คลิกลิงก์ `http://localhost:8888/...` ใน terminal) → เข้าโฟลเดอร์ `notebooks/` → เปิด `lab00_setup_check.ipynb` แล้วกด `Shift+Enter` ทีละ cell
ถ้า cell แรกพิมพ์ version ของทุก library ได้ แสดงว่าพร้อมแล้ว ไปต่อที่ lab01 ตามลำดับในตารางด้านล่าง
ปิด Jupyter ด้วย `Ctrl+C` ใน terminal (กดสองครั้ง) — งานที่ทำใน notebook ถูกบันทึกเมื่อกด save (`Ctrl+S`) ใน Jupyter

**ใช้ VS Code แทนได้:** ติดตั้ง extension "Python" และ "Jupyter" → File → Open Folder เลือก `neural-network-lab` → เปิดไฟล์ `.ipynb` → มุมขวาบน "Select Kernel" → เลือก `.venv` (Python 3.10) ที่อยู่ในโฟลเดอร์นี้

### แบบฝึกหัดใน notebook

lab01-13 ปิดท้ายด้วยส่วน **แบบฝึกหัด (ทำเอง)** ที่ต้องเขียนโค้ดจริง แต่ละข้อมี 3 cell: โจทย์ → cell โครงที่มี `raise NotImplementedError` ให้แทนที่ด้วยโค้ดของคุณ → cell ตรวจคำตอบ
cell ตรวจรันได้ทันที: ⏳ = ยังไม่ได้ทำ · ✗ = ยังไม่ถูก (มีคำใบ้) · ✓ PASS = ถูกแล้ว ทำซ้ำได้ไม่จำกัด และมี cell `summary()` สรุปท้าย lab

### อัปเดต lab เมื่อผู้สอนแก้ไข

รันจากในโฟลเดอร์ `neural-network-lab` (`cd` เข้าไปก่อนเหมือนขั้นที่ 3)

```bash
git pull
```
ถ้า git บอกว่าไฟล์ที่คุณแก้ (เช่น notebook ที่ทำแบบฝึกหัดไว้) จะถูกเขียนทับ ให้เก็บงานของคุณก่อนแล้วดึงใหม่

```bash
git stash && git pull && git stash pop
```
หรือ copy notebook ที่ทำไว้ไปตั้งชื่อใหม่ (เช่น `lab05_mywork.ipynb`) ก่อน `git pull` ไฟล์ที่ลงท้าย `_mywork.ipynb` จะไม่ถูก git ติดตาม

### ปัญหาที่พบบ่อย

| อาการ | วิธีแก้ |
|---|---|
| `uv: command not found` / `'uv' is not recognized` | ปิดแล้วเปิด terminal ใหม่ (ต้องโหลด PATH ใหม่หลังติดตั้ง) |
| `No pyproject.toml found in current directory` หรือ Jupyter เปิดแล้วไม่เห็นโฟลเดอร์ `notebooks` | terminal ไม่ได้อยู่ในโฟลเดอร์ lab — `cd ~/Documents/neural-network-lab` (Windows: `cd C:\lab\neural-network-lab`) แล้วรันใหม่ |
| `fatal: destination path 'neural-network-lab' already exists` ตอน clone | เคย clone ไว้แล้ว — ไม่ต้อง clone ซ้ำ แค่ `cd neural-network-lab` แล้วทำขั้นต่อไป (อัปเดตด้วย `git pull`) |
| Windows: `running scripts is disabled on this system` | รัน PowerShell ด้วยคำสั่งติดตั้งที่มี `-ExecutionPolicy ByPass` ตามด้านบน หรือใช้ Command Prompt |
| `ModuleNotFoundError: No module named 'nnlab'` ใน notebook | เลือก kernel ผิด — เปิด Jupyter ด้วย `uv run jupyter lab` เสมอ หรือใน VS Code เลือก kernel `.venv` |
| `uv sync` ช้าหรือหลุดกลางทาง | รันซ้ำได้เลย จะดาวน์โหลดต่อจากที่ค้าง; ถ้าใช้ Wi-Fi มหาวิทยาลัยแล้วช้า ลอง hotspot |
| lab12/lab13 ดาวน์โหลด MNIST / น้ำหนัก ResNet-18 ไม่สำเร็จ | ทำ cell ท้าย lab00 ที่บ้านล่วงหน้า หรือขอไฟล์จากผู้สอนแล้ววางตาม path ที่ README บอก |
| macOS: `xcrun: error` ตอน git | รัน `xcode-select --install` แล้วรอติดตั้งให้เสร็จ |

---

## สำหรับผู้สอน / ผู้ที่อยากปรับแต่ง

- notebook ทั้ง 14 ไฟล์รันเก็บ output ไว้แล้ว GitHub แสดงผลได้เลยโดยไม่ต้องรัน
- ไม่มี uv: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` (มี `-e .` อยู่ในไฟล์แล้ว)
- **lab13 ใช้น้ำหนัก ResNet-18 ที่ pretrained บน ImageNet** (45 MB) ซึ่ง torchvision ดาวน์โหลดอัตโนมัติครั้งแรกไปที่ `~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth` (`TORCH_HOME` เปลี่ยนตำแหน่งได้)
- **ถ้าโฟลเดอร์ lab อยู่ใน Google Drive / OneDrive ที่ sync อยู่** ให้ตั้ง `.venv` ไว้นอกโฟลเดอร์ sync: `export UV_PROJECT_ENVIRONMENT=~/.venvs/neural-network-lab` ก่อน `uv sync` (`.venv` มีไฟล์หลายหมื่นไฟล์ ~1 GB) และ `NNLAB_DATA_DIR` ย้าย `data/mnist` ได้เช่นกัน

---

## ลำดับการเรียน

| Lab | หัวข้อ | สไลด์หน้า | Dataset | เวลา |
|---|---|---|---|---|
| **A พื้นฐาน** | | | | |
| [00](notebooks/lab00_setup_check.ipynb) | ตรวจ environment | – | – | 15 นาที |
| [01](notebooks/lab01_python_to_numpy.ipynb) | Python list → numpy ndarray: shape, axis, `*` vs `@`, broadcasting, vectorization | 7-8, 35-39 | – | 90 นาที |
| [02](notebooks/lab02_math_prerequisites_with_numpy.ipynb) | คณิตศาสตร์พื้นฐาน: activation functions, derivative ด้วยการ nudge, computational graph | 2-12, 22-24 | – | 60 นาที |
| [03](notebooks/lab03_pandas_dataframe_to_matrix.ipynb) | pandas: CSV → one-hot → matrix, stratified split, normalization, สะพานข้าม convention | 71-73, 98-99, 122 | churn | 75 นาที |
| [04](notebooks/lab04_torch_tensor_vs_numpy.ipynb) | torch.Tensor เทียบ ndarray ทีละบรรทัด, autograd | 10-12, 22-24 | – | 75 นาที |
| **B perceptron** | | | | |
| [05](notebooks/lab05_perceptron_from_scratch.ipynb) | Perceptron / logistic regression จากศูนย์: forward, BCE, gradient descent, loop vs vectorized | 13-31, 40-56 | lung toy, blobs | 120 นาที |
| [06](notebooks/lab06_logistic_regression_three_ways.ipynb) | โมเดลเดียวกัน 3 วิธี: numpy / scikit-learn / PyTorch + มุมมอง MLE | 32-34, 57-70 | breast cancer | 90 นาที |
| **C neural network** | | | | |
| [07](notebooks/lab07_neural_network_from_scratch.ipynb) | Neural network L ชั้นจากศูนย์: notation, forward + cache, backward, gradient check, เทียบ PyTorch | 74-80 | moons | 120 นาที |
| [08](notebooks/lab08_regularization_and_tuning.ipynb) | Bias/variance, L2 / weight decay, dropout, early stopping, grid vs random search | 81-96, 115-120 | moons, churn | 90 นาที |
| [09](notebooks/lab09_optimizers_and_minibatch.ipynb) | Input normalization, mini-batch, Momentum / RMSprop / Adam, learning-rate decay | 97-114 | churn, breast cancer | 90 นาที |
| **D evaluation** | | | | |
| [10](notebooks/lab10_experiment_and_evaluation.ipynb) | train/val/test, k-fold, confusion matrix, precision/recall/F1, ROC, micro vs macro | 121-152 | churn, breast cancer, digits | 90 นาที |
| **E CNN** | | | | |
| [11](notebooks/lab11_convolution_from_scratch.ipynb) | Convolution จากศูนย์: ภาพเป็น matrix, filter, padding, stride, volume, pooling, เทียบ `F.conv2d` | 153-198 | digits, MNIST 1 ภาพ | 120 นาที |
| [12](notebooks/lab12_cnn_with_pytorch.ipynb) | LeNet-5 ด้วย PyTorch บน MNIST: DataLoader, training loop, filters, augmentation, classic CNNs | 199-210 | MNIST | 120 นาที |
| [13](notebooks/lab13_cnn_in_practice.ipynb) | CNN ในงานจริง: custom image `Dataset` จากไฟล์ + CSV, augmentation ตอนเทรน, early stopping + checkpoint, transfer learning ด้วย ResNet-18 (freeze → head → fine-tune) | 96, 122-125, 203-210 | cifar3 (bird/dog/frog) | 120 นาที |

ทุก notebook มีโครงเดียวกัน: จุดประสงค์ → ขั้นที่ 1-4 (ปัญหา → แนวคิด → ทฤษฎี → สาธิตด้วยตัวเลขจากสไลด์) →
ขั้นสุดท้าย เวอร์ชัน production ใน `nnlab` และ **numpy | torch เทียบบรรทัดต่อบรรทัด** → ★ Key Takeaways → ลองทำเอง

---

## Convention ของข้อมูล — จุดที่สับสนบ่อยที่สุด

| | สไลด์ (Andrew Ng) | pandas / scikit-learn / PyTorch |
|---|---|---|
| `X` | `(n_x, m)` sample เป็น**คอลัมน์** | `(m, n_x)` sample เป็น**แถว** |
| label | `Y` `(1, m)` | `y` `(m,)` |
| linear step | `Z = W @ X + b` | `Z = X @ W.T + b` |
| `W[l]` | `(n[l], n[l-1])` | `nn.Linear(n_in, n_out).weight` = `(n_out, n_in)` — shape เดียวกัน |

notebook สอน convention ของสไลด์ก่อนในส่วน numpy จากศูนย์ แล้วมี cell "สะพาน" (`nnlab.conventions.to_deck / to_lib`) ทุกครั้งที่สลับ
**public API ของ `nnlab` ทุกโมเดลรับ `X (m, n_x)`** แบบ library และ transpose ภายในเอง

---

## โครงสร้างโฟลเดอร์

```
lab/
├── notebooks/          lab00 … lab13 (.ipynb รันเก็บ output แล้ว, ท้ายไฟล์มีแบบฝึกหัดพร้อม cell ตรวจคำตอบ)
├── src/nnlab/          package: conventions, activations, losses, perceptron, nn, optimizers,
│                       metrics, conv, torch_models, baselines, data, preprocessing, plotting, config, experiment, utils, vision
├── scripts/            CLI: make_data.py, make_image_dataset.py, train.py, evaluate.py, compare_frameworks.py, train_cnn.py, train_transfer.py
├── configs/            ตัวอย่างไฟล์ config JSON
├── data/               CSV เล็ก (สร้างซ้ำได้) + cifar3/ (1,800 PNG ≈ 5 MB commit ไว้) + mnist/ (ดาวน์โหลดอัตโนมัติ, ไม่เก็บใน git)
├── tests/              pytest ทุกโมดูล (ค่าคาดหวังมาจากตัวเลขในสไลด์) + test_notebooks.py
├── tools/              ตัวสร้าง notebook (nbformat) และ exercise_check.py สำหรับตรวจว่าแบบฝึกหัดทำได้จริงด้วยเฉลย
├── runs/               ผลลัพธ์ของ scripts (ไม่เก็บใน git)
├── pyproject.toml, uv.lock, requirements.txt
└── Perceptron, Neural Network, and Convolutional Neural Network.pdf   ← สไลด์บรรยาย (ฉบับแก้ไข 10 ก.ย. 2026)
```

ทุกโมเดลใน `nnlab` (numpy `Perceptron`, `NeuralNetwork` · sklearn `SklearnLogReg`, `SklearnMLP` · torch `TorchTrainer`)
ใช้ interface เดียวกัน `fit / predict_proba / predict / evaluate / save / load` และมี `history_` จึงสลับกันได้ใน script

---

## Scripts (ฝั่ง production)

```bash
uv run python scripts/make_data.py                       # สร้าง data/*.csv ใหม่ (deterministic) ; --check ตรวจอย่างเดียว
uv run python scripts/train.py --model numpy-perceptron --dataset breast_cancer --epochs 300
uv run python scripts/train.py --model numpy-mlp --dataset moons --hidden 16 8 --optimizer adam --lr 0.01 --epochs 500
uv run python scripts/train.py --config configs/example_train.json --epochs 200      # config file + override
uv run python scripts/evaluate.py --latest --threshold 0.4                         # confusion matrix, metrics, roc.png
uv run python scripts/compare_frameworks.py                                        # numpy vs sklearn vs torch ตารางเดียว
uv run python scripts/train_cnn.py --epochs 3 --limit-train 10000                  # LeNet-5 บน MNIST (~1 นาที CPU)
uv run python scripts/train_cnn.py --dataset digits --epochs 10                    # offline ไม่ต้องดาวน์โหลด
uv run python scripts/train_cnn.py --epochs 20 --limit-train 5000 --patience 2     # early stopping บน val 10% ของ train
uv run python scripts/train_transfer.py --mode all --fast                          # scratch vs head vs fine-tune บน cifar3 (~1 นาที)
uv run python scripts/train_transfer.py --mode finetune --epochs 8 --patience 3    # ResNet-18: freeze → head → fine-tune layer4
uv run python scripts/make_image_dataset.py --check                                # ตรวจ data/cifar3 (regenerate ต้องดาวน์โหลด CIFAR-10 170 MB)
```
โมเดลที่ `train.py` รู้จัก: `numpy-perceptron | numpy-mlp | sklearn-logreg | sklearn-mlp | torch-logreg | torch-mlp`
dataset: `lung | churn | breast_cancer | moons | blobs | digits` — ผลทุกอย่างอยู่ใน `runs/<timestamp>_<model>_<dataset>/`

## Tests

```bash
uv run pytest                    # ~1 นาที: ทุกโมดูล (ตัวเลขจากสไลด์เป็นค่าคาดหวัง)
uv run pytest -m slow            # execute ทุก notebook ตั้งแต่ต้นจนจบ (หลายนาที)
```

---

## หมายเหตุสำหรับผู้ที่มีสไลด์ฉบับก่อน 10 ก.ย. 2026

สไลด์ในโฟลเดอร์นี้เป็นฉบับแก้ไขแล้ว ถ้าใช้ฉบับเก่าจะพบตัวเลขต่างจาก notebook ในจุดเหล่านี้ (notebook คำนวณค่าที่ถูกให้ดูสด)
- p.52, 60, 63: σ(z) ของตัวอย่างมะเร็งปอดคือ [0.690, 0.690, 0.668] (ฉบับเก่าพิมพ์ [0.49, 0.45, 0.45])
- p.148-151: macro average ต้องเฉลี่ย metric ต่อ class — precision 0.807, recall 0.837, F1 0.813
- p.144: TN ของ class B = 28 · p.199-200: CONV2 = 10×10×16 และตาราง CONV1/POOL1 ใช้ 6 channel
- p.49: diagram เขียน sample ที่ 1 ไว้ขวาสุด จึงอ่าน z ได้ [0.7, 0.8, 0.8] ส่วนโค้ดเรียง sample ที่ 1 ไว้คอลัมน์แรก [0.8, 0.8, 0.7] — ตัวเลขชุดเดียวกัน

---

## สำหรับผู้สอน

- notebook ถูกรันเก็บ output ไว้แล้ว ถ้าแก้โค้ดใน `src/nnlab/` ให้รันใหม่ด้วย
  `uv run jupyter nbconvert --to notebook --execute --inplace notebooks/labNN_*.ipynb`
- notebook ทุกไฟล์สร้างจาก script ใน `tools/notebook_sources/` (ใช้ `nbformat`) — แก้ `.ipynb` ตรงๆ ใน Jupyter ได้ หรือแก้ script แล้วรัน
  `uv run python tools/notebook_sources/lab05.py` เพื่อสร้างใหม่ทั้งไฟล์ (คู่มือสไตล์อยู่ที่ `tools/notebook_sources/NOTEBOOK_GUIDE.md`)
- `scripts/make_data.py --check` ใช้ยืนยันว่า CSV ยังตรงกับตัวสร้าง (มีอยู่ใน pytest แล้ว)
- ตัวเลขทุกตัวในสไลด์ที่ notebook อ้าง ถูกยืนยันด้วย test ใน `tests/` (เช่น `test_conv.py`, `test_metrics.py`, `test_perceptron.py`)
