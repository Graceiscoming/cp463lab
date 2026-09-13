import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from nb_builder import NB

nb = NB()
nb.header(1, "จาก Python list สู่ numpy ndarray", "Python lists → numpy arrays: shape, axis, broadcasting, vectorization",
    objectives=["อธิบายได้ว่า `ndarray` ต่างจาก `list` อย่างไร และอ่าน `shape`, `dtype`, `ndim` ได้",
                "ใช้ indexing / slicing / `reshape` / `.T` และเข้าใจความหมายของ `axis` กับ `keepdims`",
                "แยกความต่างของ `*` (elementwise / Hadamard) กับ `@` (dot product / matrix multiplication)",
                "ทำนายผลของ broadcasting ได้ทั้ง 4 กรณีในสไลด์ p.35-38 รวมกรณีที่ error",
                "เขียนสมการหลายตัวแปรเป็น matrix (p.39) และเห็นว่า vectorization เร็วกว่า loop กี่เท่า"],
    slides="7-8, 35-39", minutes=90)
nb.setup()
nb.md("""
## สัญลักษณ์ในบทนี้
| สัญลักษณ์ | ความหมาย | ใน numpy |
|---|---|---|
| $u \\cdot v$ | dot product ของ vector (p.7) | `u @ v` หรือ `np.dot(u, v)` |
| $A \\odot B$ | pairwise (Hadamard) multiplication (p.8) | `A * B` |
| $A^T$ | transpose สลับแถวกับคอลัมน์ | `A.T` |
| $\\mathbb{R}^{n \\times m}$ | matrix n แถว m คอลัมน์ | `A.shape == (n, m)` |
""")

nb.md("""
## ขั้นที่ 1 · ทำไม list ของ Python ไม่พอ
สมมติเรามีคะแนนของนิสิต 3 คนและอยากบวก 5 ให้ทุกคน กับ list เราต้องวน loop เอง
และถ้าเผลอเขียน `scores + 5` Python จะ error เพราะ list ไม่รู้จักการบวกกับตัวเลข
ลองดูว่าเกิดอะไรขึ้น (เราดัก error ไว้เพื่อให้ cell รันต่อได้)
""")
nb.code("""
scores = [70, 85, 90]
try:
    scores + 5
except TypeError as e:
    print("list + 5 →", type(e).__name__, ":", e)

plus_five = [s + 5 for s in scores]          # ต้องวน loop เอง
print("วน loop เอง:", plus_five)
print("list * 2  :", scores * 2, "  ← ไม่ใช่การคูณตัวเลข แต่เป็นการซ้ำ list!")
""")
nb.md("""
`numpy.ndarray` (มักเรียกสั้นๆ ว่า array) เก็บตัวเลข **ชนิดเดียวกัน** เรียงต่อกันในหน่วยความจำ
จึงบวก/คูณทั้งก้อนได้ในคำสั่งเดียว (เรียกว่า **vectorization**) และเร็วกว่า loop มาก
array มีคุณสมบัติสำคัญ 3 อย่างที่เราจะดูตลอดคอร์ส: `shape` (ขนาดแต่ละมิติ), `dtype` (ชนิดตัวเลข), `ndim` (จำนวนมิติ)
""")
nb.code("""
arr = np.array(scores)
print("arr      :", arr)
print("arr + 5  :", arr + 5)                 # บวกทุกตัวในคำสั่งเดียว
print("arr * 2  :", arr * 2)                 # คูณตัวเลขจริงๆ
print("shape:", arr.shape, "| dtype:", arr.dtype, "| ndim:", arr.ndim)

arr_f = np.array([70, 85, 90], dtype=float)  # ระบุชนิดเป็นทศนิยม (float64)
print("dtype float:", arr_f.dtype, arr_f)
""")

nb.md("""
## ขั้นที่ 2 · vector, matrix และ shape
- **vector** = array 1 มิติ `shape (n,)` — ในสไลด์ p.7 $u = (3, 4)$
- **matrix** = array 2 มิติ `shape (แถว, คอลัมน์)` — ในสไลด์ p.8 "a stack of vectors"

ใน deep learning เราจะเจอ **column vector** `shape (n, 1)` และ **row vector** `shape (1, n)` บ่อยมาก
ทั้งสามแบบ `(3,)`, `(3, 1)`, `(1, 3)` เก็บตัวเลข 3 ตัวเหมือนกัน แต่ **พฤติกรรมตอนคูณและ broadcast ต่างกัน**
นี่คือต้นเหตุของ bug ส่วนใหญ่ในการเขียน neural network ด้วยมือ
""")
nb.code("""
u = np.array([3, 4])                 # vector 1 มิติ
A = np.array([[1, 2, 3],
              [4, 5, 6]])            # matrix 2 แถว 3 คอลัมน์ (สไลด์ p.8)

print("u:", u, "shape", u.shape, "ndim", u.ndim)
print("A:\\n", A, "\\nshape", A.shape, "ndim", A.ndim)

col = np.array([[3], [4]])           # column vector (2, 1)
row = np.array([[3, 4]])             # row vector (1, 2)
print("\\ncolumn vector shape", col.shape, "| row vector shape", row.shape)
""")
nb.md("""
### indexing และ slicing
`A[i, j]` = แถว i คอลัมน์ j (นับจาก 0) · `A[i, :]` = ทั้งแถว i · `A[:, j]` = ทั้งคอลัมน์ j
สังเกตว่าการดึงทั้งคอลัมน์ได้ array 1 มิติ `(2,)` ไม่ใช่ `(2, 1)` — numpy "บีบ" มิติที่เหลือ 1 ทิ้ง
ถ้าอยากคง 2 มิติไว้ ใช้ slice `A[:, 0:1]`
""")
nb.code("""
print("A[0, 2]  =", A[0, 2])            # แถว 0 คอลัมน์ 2 → 3
print("A[1, :]  =", A[1, :], A[1, :].shape)      # ทั้งแถว 1
print("A[:, 0]  =", A[:, 0], A[:, 0].shape)      # ทั้งคอลัมน์ 0 → 1 มิติ!
print("A[:, 0:1]=\\n", A[:, 0:1], A[:, 0:1].shape)  # slice → คง 2 มิติ (2, 1)
print("A[:, -1] =", A[:, -1], " ← คอลัมน์สุดท้าย")
""")
nb.md("""
### reshape และ transpose
`reshape` เปลี่ยนรูปร่างโดยจำนวนตัวเลขต้องเท่าเดิม (`-1` = ให้ numpy คำนวณมิตินั้นเอง) — ใช้แปลง `(3,)` ↔ `(3, 1)` ↔ `(1, 3)`
`.T` (transpose) สลับแถวกับคอลัมน์ ตามสไลด์ p.8: $A^T$ ของ matrix 2×3 คือ 3×2
""")
nb.code("""
v = np.array([1, 2, 3])
print("v.reshape(3, 1):\\n", v.reshape(3, 1))          # column vector
print("v.reshape(1, -1):", v.reshape(1, -1), v.reshape(1, -1).shape)   # -1 ให้คำนวณเอง → (1, 3)
print("\\nA.T:\\n", A.T, "\\nshape", A.shape, "→", A.T.shape)
print("\\ntranspose ของ vector 1 มิติไม่เปลี่ยนอะไร:", v.T.shape, "← ต้อง reshape ก่อนถ้าอยากได้ column")
""")

nb.md("""
## ขั้นที่ 3 · axis และ keepdims
`A.sum()` รวมทุกตัว แต่บ่อยครั้งเราอยากรวม "ตามแนว" — นี่คือ `axis`
- `axis=0` = ยุบมิติที่ 0 (แถว) → ได้ผลรวม **ของแต่ละคอลัมน์**
- `axis=1` = ยุบมิติที่ 1 (คอลัมน์) → ได้ผลรวม **ของแต่ละแถว**

วิธีจำ: axis ที่ระบุคือมิติที่ **หายไป** จาก shape ผลลัพธ์
`keepdims=True` ทำให้มิตินั้นเหลือขนาด 1 แทนที่จะหายไป — จำเป็นตอน broadcast กลับ (จะเห็นใน lab07 `db = np.sum(dZ, axis=1, keepdims=True)`)
""")
nb.code("""
print("A =\\n", A)
print("A.sum()          =", A.sum())
print("A.sum(axis=0)    =", A.sum(axis=0), A.sum(axis=0).shape, " ← รวมแต่ละคอลัมน์ มิติ 0 หาย")
print("A.sum(axis=1)    =", A.sum(axis=1), A.sum(axis=1).shape, " ← รวมแต่ละแถว มิติ 1 หาย")
print("A.sum(axis=1, keepdims=True) =\\n", A.sum(axis=1, keepdims=True), A.sum(axis=1, keepdims=True).shape)
print("\\nmean แต่ละคอลัมน์:", A.mean(axis=0), "| max แต่ละแถว:", A.max(axis=1))
""")

nb.md("""
## ขั้นที่ 4 · `*` กับ `@` ไม่เหมือนกัน
สไลด์ p.7-8 มีการคูณ 2 แบบที่ต้องแยกให้ชัด
- **pairwise (Hadamard) multiplication** $A \\odot B$: คูณตำแหน่งต่อตำแหน่ง ต้อง shape เท่ากัน → numpy `A * B`
- **dot product / matrix multiplication** $u \\cdot v$, $A \\cdot B$: คูณแล้วรวม → numpy `A @ B` (หรือ `np.dot`)
  กติกา: `(n, k) @ (k, m) → (n, m)` มิติกลางต้องเท่ากัน

ตัวอย่างสไลด์ p.7: $u = (3, 4)$, $v = (4, 3)$ → $u \\cdot v = 3 \\times 4 + 4 \\times 3 = 24$ และความยาว $\\|u\\| = \\sqrt{3^2 + 4^2} = 5$
""")
nb.code("""
u = np.array([3, 4]); v = np.array([4, 3])
print("u * v =", u * v, "  ← pairwise")
print("u @ v =", u @ v, "  ← dot product = 3×4 + 4×3 = 24 (สไลด์ p.7)")
print("ความยาว ‖u‖ = sqrt(u @ u) =", np.sqrt(u @ u), "= np.linalg.norm(u) =", np.linalg.norm(u))

B = np.full((2, 3), 2)                       # matrix ของเลข 2 ทั้งหมด (สไลด์ p.8)
print("\\nA ⊙ B = A * B =\\n", A * B)

print("\\nA @ A.T =\\n", A @ A.T, " shape (2,3)@(3,2) → (2,2)")
try:
    A @ A                                     # (2,3)@(2,3) มิติกลาง 3 ≠ 2
except ValueError as e:
    print("\\nA @ A → ValueError:", e)
""")

nb.md("""
## ขั้นที่ 5 · broadcasting (สไลด์ p.35-38)
broadcasting คือกฎที่ numpy ใช้ "ยืด" array ที่เล็กกว่าให้มี shape เท่ากันก่อนคำนวณ โดย**ไม่ต้อง copy ข้อมูลจริง**
กติกา: เทียบ shape จากขวาไปซ้าย ทีละมิติ — ใช้ได้ถ้าเท่ากัน **หรือ** ฝ่ายใดฝ่ายหนึ่งเป็น 1 (หรือไม่มีมิตินั้น)
เรามาทำซ้ำ 4 กรณีในสไลด์ทีละกรณี
""")
nb.code("""
# กรณี 1 (p.35): vector + scalar → scalar ถูก replicate ให้ทุกตำแหน่ง
v = np.array([1, 2, 3])
print("v + 3 =", v + 3)

# กรณี 2 (p.36): matrix + scalar
V = np.array([[1, 2, 3], [4, 5, 6]])
print("V + 3 =\\n", V + 3)

# กรณี 3 (p.37): matrix (2,3) + vector (3,) → vector ถูก replicate ลงทุกแถว
u = np.array([10, 20, 30])
print("V + u =\\n", V + u, "   shape (2,3) + (3,) → (2,3)")
""")
nb.md("""
กรณี 4 (สไลด์ p.38): matrix `(2, 3)` + vector `(2,)` — เทียบจากขวา: 3 กับ 2 ไม่เท่ากันและไม่มีใครเป็น 1 → **error**
ถ้าตั้งใจจะบวก `[10, 20]` ให้แถวที่ 0 และ 1 ตามลำดับ ต้องทำให้เป็น column vector `(2, 1)` ก่อน แล้ว broadcast จะยืดไปตามคอลัมน์
""")
nb.code("""
u2 = np.array([10, 20])
try:
    V + u2
except ValueError as e:
    print("V (2,3) + u2 (2,) → ValueError:", e)

print("\\nแก้ด้วย reshape เป็น (2,1):\\n", V + u2.reshape(2, 1), "   (2,3) + (2,1) → (2,3)")
print("\\nตารางสรุป: (2,3)+(3,) ✓ | (2,3)+(1,3) ✓ | (2,3)+(2,1) ✓ | (2,3)+(2,) ✗")
""")
nb.md("""
เราจะใช้ broadcasting ตลอดคอร์สโดยเฉพาะ `Z = W @ X + b` ที่ `b` มี shape `(n, 1)` แต่ `W @ X` มี `(n, m)`
— numpy ยืด `b` ไปทุกคอลัมน์ (ทุก sample) ให้เอง นี่คือเหตุผลที่สไลด์ p.28 บอกว่า "broadcasting enables implementation without looping over training data"
""")

nb.md("""
## ขั้นที่ 6 · เขียนระบบสมการเป็น matrix (สไลด์ p.39)
$$2X_1 - 4X_2 + 6X_3 = 10,\\quad -X_1 + 2X_2 + 4X_3 = 12,\\quad 5X_1 + 7X_2 - 2X_3 = 15$$
เขียนได้เป็น $\\Theta X = Z$ โดย $\\Theta$ คือ matrix สัมประสิทธิ์ 3×3, $X$ คือ column vector ของตัวแปร, $Z$ คือ column vector ทางขวา
ถ้ารู้ $X$ ก็หา $Z$ ด้วย `Theta @ X` และถ้ารู้ $Z$ ก็หา $X$ ด้วย `np.linalg.solve` — แนวคิด "สมการหลายตัว = การคูณ matrix ครั้งเดียว" นี่เองที่ทำให้ perceptron หลาย sample คำนวณพร้อมกันได้
""")
nb.code("""
Theta = np.array([[2, -4, 6],
                  [-1, 2, 4],
                  [5, 7, -2]], dtype=float)
Z = np.array([[10], [12], [15]], dtype=float)         # column vector (3, 1)

X = np.linalg.solve(Theta, Z)                          # หา X ที่ทำให้ Theta @ X = Z
print("X =\\n", X)
print("ตรวจ Theta @ X =\\n", Theta @ X, "\\nตรงกับ Z:", np.allclose(Theta @ X, Z))
""")

nb.md("""
## ขั้นที่ 7 · vectorization เร็วกว่า loop แค่ไหน
สไลด์ p.29-31 เทียบ "naive implementation" (loop) กับ "vectorized" เราจะวัดจริงด้วย dot product ของ vector ยาว 1 ล้านตัว
`%timeit` เป็นคำสั่งพิเศษของ Jupyter ที่รันซ้ำหลายรอบแล้วรายงานเวลาเฉลี่ย
""")
nb.code("""
n = 1_000_000
a = rng.random(n); b = rng.random(n)

def dot_loop(a, b):
    total = 0.0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

assert np.isclose(dot_loop(a, b), a @ b)       # ผลเท่ากัน
print("loop      :", end=" "); t_loop = %timeit -o -q dot_loop(a, b)
print("vectorized:", end=" "); t_vec = %timeit -o -q a @ b
print(f"\\nloop {t_loop.average*1e3:.1f} ms  vs  vectorized {t_vec.average*1e3:.3f} ms  → เร็วกว่า ~{t_loop.average / t_vec.average:,.0f} เท่า")
""")

nb.production_note("conventions.py", "ฟังก์ชัน `describe` ที่บอกความหมายของ shape", step=8)
nb.code("""
from nnlab.conventions import describe

X_rows = np.array([[3, 1], [5, 0], [2, 1]], dtype=float)     # 3 sample × 2 feature (ตารางสไลด์ p.40)
print("library convention:", describe(X_rows, "lib"))
print("deck convention   :", describe(X_rows.T, "deck"))
""")
nb.takeaways([
    "`ndarray` เก็บตัวเลขชนิดเดียวกัน คำนวณทั้งก้อนได้ในคำสั่งเดียว (vectorization) และเร็วกว่า loop หลายร้อยเท่า",
    "อ่าน `shape` ให้ติดเป็นนิสัย — `(3,)`, `(3, 1)`, `(1, 3)` เก็บเลข 3 ตัวเหมือนกันแต่คูณและ broadcast ต่างกัน",
    "`axis` คือมิติที่ถูกยุบหาย; `keepdims=True` เก็บมิตินั้นไว้เป็นขนาด 1 เพื่อ broadcast กลับได้",
    "`*` = pairwise (Hadamard) · `@` = dot / matrix multiplication ที่มิติกลางต้องเท่ากัน",
    "broadcasting เทียบ shape จากขวาไปซ้าย: เท่ากันหรือเป็น 1 จึงใช้ได้ — `(2,3)+(2,)` จึง error แต่ `(2,3)+(2,1)` ได้",
    "ระบบสมการหลายตัว = การคูณ matrix ครั้งเดียว — รากฐานของการคำนวณหลาย sample พร้อมกันใน lab05",
])
nb.exercises_intro(1)
nb.exercise("1.1", "สถิติต่อแกนโดยไม่ใช้ loop",
    goal="ใช้ `axis` หา mean ของแต่ละคอลัมน์ และ max ของแต่ละแถวของ matrix สุ่ม 4×3",
    steps=["`M` ถูกสร้างให้แล้วใน cell โครง (shape `(4, 3)`) — พิมพ์ดูก่อน",
           "กำหนด `col_means` = ค่าเฉลี่ยของแต่ละคอลัมน์ → ต้องได้ shape `(3,)`",
           "กำหนด `row_max` = ค่ามากสุดของแต่ละแถว → ต้องได้ shape `(4,)`",
           "ห้ามใช้ `for` — ใช้ `axis=` เท่านั้น (ขั้นที่ 3)"],
    skeleton="""
    M = rng.integers(0, 10, size=(4, 3))        # M: (4, 3)
    print(M)
    col_means = ...                              # TODO: mean ของแต่ละคอลัมน์ (3,)   ← แทน ... ด้วยโค้ดของคุณ
    row_max = ...                                # TODO: max ของแต่ละแถว (4,)
    print("col_means =", col_means, "| row_max =", row_max)
    """,
    check_code="""
    # เทียบกับค่าที่คำนวณด้วย loop (คนละวิธี)
    ref_means = [sum(M[i, j] for i in range(4)) / 4 for j in range(3)]
    ref_max = [max(M[i, j] for j in range(3)) for i in range(4)]
    check_shape("1.1 col_means shape (3,)", lambda: col_means, (3,), hint="mean ตามแกนที่ยุบแถวทิ้ง")
    check_close("1.1 col_means ค่าถูก", lambda: col_means, ref_means, hint="axis=0 คือยุบมิติแถว → เหลือค่าต่อคอลัมน์")
    check_close("1.1 row_max ค่าถูก", lambda: row_max, ref_max, hint="M.max(axis=?) — มิติที่หายคือคอลัมน์")
    """,
    hints=["`M.mean(axis=0)` ยุบมิติ 0 (แถว) ทิ้ง เหลือ 1 ค่าต่อคอลัมน์", "`max` ก็รับ `axis` เหมือน `sum`/`mean`"])
nb.exercise("1.2", "standardize ทุกคอลัมน์ด้วย broadcasting",
    goal="เขียนฟังก์ชันที่ทำให้แต่ละคอลัมน์ของ matrix มี mean 0 และ std 1 โดยไม่ใช้ loop (สูตร `(x - μ) / σ` ต่อคอลัมน์ — จะได้ใช้จริงใน lab03)",
    steps=["คำนวณ `mu` และ `sigma` ของแต่ละคอลัมน์ด้วย `axis=0` → shape `(n_col,)`",
           "คืน `(X - mu) / sigma` — broadcasting จะกระจาย `mu` shape `(n_col,)` ให้ทุกแถวเอง (กรณี matrix + vector ในขั้นที่ 5)",
           "ทดสอบกับ `X = rng.normal(5, 2, size=(50, 3))` แล้วดูว่า mean แต่ละคอลัมน์ ≈ 0"],
    skeleton="""
    def standardize_columns(X):
        \"\"\"คืน matrix ที่แต่ละคอลัมน์มี mean 0, std 1 (X: (m, n))\"\"\"
        raise NotImplementedError("ยังไม่ได้ทำ")

    X_test = rng.normal(5, 2, size=(50, 3))
    # Z = standardize_columns(X_test); print(Z.mean(axis=0), Z.std(axis=0))
    """,
    check_code="""
    check_shape("1.2 shape เท่าเดิม", lambda: standardize_columns(X_test), (50, 3))
    check_close("1.2 mean ทุกคอลัมน์ ≈ 0", lambda: standardize_columns(X_test).mean(axis=0), [0, 0, 0], atol=1e-9, hint="ลบ mu ที่คำนวณด้วย axis=0")
    check_close("1.2 std ทุกคอลัมน์ ≈ 1", lambda: standardize_columns(X_test).std(axis=0), [1, 1, 1], atol=1e-9, hint="หารด้วย std ของแต่ละคอลัมน์ (axis=0) ไม่ใช่ std รวม")
    check("1.2 ไม่เปลี่ยน X เดิม (ห้ามแก้ in-place)", lambda: standardize_columns(X_test) is not None and abs(X_test.mean() - 5) < 1, hint="สร้าง array ใหม่ด้วย (X - mu) / sigma แทนการทำ X -= mu")
    """,
    hints=["`mu = X.mean(axis=0)` มี shape `(3,)` และ `X - mu` ใช้ได้ทันทีเพราะเทียบจากขวา 3 == 3", "ถ้าเขียน `X -= mu` จะแก้ array ต้นทาง — ให้คืน array ใหม่"])
nb.exercise("1.3", "outer product ด้วย loop สองชั้น เทียบกับ numpy",
    goal="เขียน `outer_loop(u, v)` ที่สร้าง matrix `M[i, j] = u[i] * v[j]` ด้วย loop สองชั้น แล้วเทียบผลและเวลากับ `np.outer`",
    steps=["สร้าง matrix ศูนย์ shape `(len(u), len(v))` ด้วย `np.zeros`",
           "วน `i` ทุกตำแหน่งของ `u` และ `j` ทุกตำแหน่งของ `v` แล้วใส่ค่า `u[i] * v[j]`",
           "คืน matrix แล้วลอง `%timeit outer_loop(a, b)` กับ `%timeit np.outer(a, b)` สำหรับ vector ยาว 300 (เพิ่ม cell เอง)"],
    skeleton="""
    def outer_loop(u, v):
        \"\"\"M[i, j] = u[i] * v[j]  → shape (len(u), len(v))\"\"\"
        raise NotImplementedError("ยังไม่ได้ทำ")
    """,
    check_code="""
    u_t, v_t = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0])
    check_shape("1.3 shape (3, 2)", lambda: outer_loop(u_t, v_t), (3, 2), hint="แถว = len(u), คอลัมน์ = len(v)")
    check_close("1.3 ค่าตรงกับ np.outer", lambda: outer_loop(u_t, v_t), np.outer(u_t, v_t), hint="M[i, j] = u[i] * v[j] — ตรวจว่า index ไม่สลับกัน")
    check_close("1.3 ใช้กับ vector ยาว 50 ได้", lambda: outer_loop(rng.random(50), rng.random(40)).shape, [50, 40])
    """,
    hints=["`M = np.zeros((len(u), len(v)))` แล้ว `for i in range(len(u)): for j in range(len(v)): M[i, j] = u[i] * v[j]`",
           "np.outer เร็วกว่าหลายร้อยเท่าเพราะ vectorized — นี่คือประเด็นของขั้นที่ 7"])
nb.exercises_summary()
nb.save("lab01_python_to_numpy.ipynb")
