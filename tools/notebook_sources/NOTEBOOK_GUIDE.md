# คู่มือเขียน notebook สำหรับ CP463 lab (ใช้กับทุกไฟล์)

## ภาษาและสำนวน
- ภาษาไทยสำหรับคำอธิบายทั้งหมด **ทับศัพท์เทคนิคภาษาอังกฤษโดยไม่แปลและไม่วงเล็บ**: gradient descent, loss function, cost function,
  weight, bias, activation function, sigmoid, ReLU, forward propagation, backward propagation, tensor, broadcasting, shape, dtype,
  axis, mini-batch, epoch, learning rate, overfitting, regularization, dropout, confusion matrix, precision, recall, convolution, filter, stride, padding, pooling
- ศัพท์ที่มีคำไทยใช้ทั่วไปแล้วใช้ไทยได้: ข้อมูล, ตัวแปร, ฟังก์ชัน, เมทริกซ์ (หรือ matrix ก็ได้), แถว/คอลัมน์, ค่าเฉลี่ย, ความน่าจะเป็น, อนุพันธ์
- เขียนเหมือนอาจารย์อธิบายในห้อง: ประโยคสั้น มีเหตุผล "ทำไม" ก่อน "อย่างไร" ยกตัวอย่างตัวเลขจริงจาก cell
- คำเรียกผู้เรียน: "นิสิต" หรือ "เรา" ; ไม่ใช้ "คุณ"
- ตัวเลขในสไลด์ที่ผิด (ดู errata) ให้ notebook คำนวณสดแล้วเขียนหมายเหตุสั้นๆ ว่าค่าที่ถูกคืออะไร ห้าม hardcode ค่าผิด
- อ้างอิงสไลด์ด้วยเลขหน้า เช่น "(สไลด์ p.30)"

## โครงทุก notebook (ใช้ NB จาก nb_builder.py)
1. `nb.header(...)` — เลข lab, ชื่อไทย, ชื่ออังกฤษ, จุดประสงค์ 3-5 ข้อ, หน้าสไลด์, เวลา, dataset
2. `nb.setup(extra=...)` — import และ seed
3. ส่วน "สัญลักษณ์ในบทนี้" (markdown ตารางสั้น) ถ้ามี notation ใหม่
4. ขั้นที่ 1 ที่มาของปัญหา → ขั้นที่ 2-3 แนวคิด/ทฤษฎี → ขั้นที่ 4 สาธิตด้วยมือ (ทำซ้ำตัวเลขสไลด์)
   - **ทุก code cell ต้องมี markdown cell นำหน้า** อธิบายว่า cell นี้ทำอะไร ทำไม และคาดว่าจะเห็นอะไร
   - ใน code cell ให้ comment shape กำกับ matrix ทุกตัว เช่น `# Z: (1, m)`
   - ใช้ `print()` แสดงค่าจริง ไม่ปล่อยให้ cell เงียบ; ใช้ `assert np.allclose(...)` เมื่อเทียบสองวิธี
5. `nb.convention("deck")` หรือ `nb.convention("lib")` ทุกครั้งที่สลับ convention ของ X
6. ตั้งแต่ lab04 เป็นต้นไป: มีส่วน **"numpy | torch เทียบบรรทัดต่อบรรทัด"** — สอง cell ติดกัน ทำงานเดียวกัน แล้ว `np.allclose` เทียบ
7. `nb.production_note(module, what)` + code cell ที่ `from nnlab... import ...` แล้วเทียบกับที่เขียนเอง
8. `nb.takeaways([...])` 4-6 ข้อ
9. `nb.exercises([...])` 2-3 ข้อ ไม่มีเฉลย

## กติกาโค้ด
- ทุก cell ต้องรันผ่านตั้งแต่ต้นจนจบด้วย `uv run jupyter nbconvert --to notebook --execute --inplace <file>` (timeout 600 s)
- เวลารันทั้ง notebook ไม่เกิน ~3 นาทีบน CPU (ลด epoch/ขนาดข้อมูล; ใช้ `os.environ.get("NNLAB_FAST")` ลดเพิ่มถ้าตั้งไว้)
- ห้ามดาวน์โหลดอะไรนอกจาก MNIST ผ่าน `nnlab.data.load_mnist` (lab11 ใช้ 1 ภาพ, lab12 ใช้ limit) และน้ำหนัก ResNet-18 ผ่าน `nnlab.vision.build_resnet18_transfer(weights="DEFAULT")` (lab13, 45 MB ไป torch cache)
- ใช้ `rng` (numpy Generator จาก set_seed) แทน `np.random.*` ; torch ใช้ `torch.manual_seed(463)`
- กราฟ: label แกนเป็นภาษาอังกฤษ (matplotlib ไม่มี font ไทย) แต่ title/คำอธิบายใน markdown เป็นไทยได้
- ห้ามแก้ไฟล์ใน `src/nnlab/` — ถ้าพบ bug ให้เขียน workaround ใน notebook และรายงาน
- `import torch` เฉพาะ lab ที่ต้องใช้ (04, 06, 07, 08, 09, 11, 12)

## errata ของสไลด์ (ต้องคำนวณสด)
- p.49: ลำดับ [0.7, 0.8, 0.8] ไม่ใช่ข้อผิดพลาด (diagram เขียน sample ที่ 1 ขวาสุด; โค้ดได้ [0.8, 0.8, 0.7]) · p.52, 60, 63 ฉบับก่อน ก.ย. 2026: σ(z) = [0.690, 0.690, 0.668] ไม่ใช่ [0.49, 0.45, 0.45] (ผู้สอนแก้แล้ว)
- p.148-151: หน้า "Macro Average" ใช้สูตร micro ซ้ำ; macro precision จริง = mean(16/17, 14/15, 6/11) = 0.807, macro F1 = 0.813
- p.199-200: CONV2 ต้องเป็น 10×10×16 (ไม่ใช่ ×10); ตาราง p.200 ใช้ 8 channel แต่ diagram ใช้ 6 — ใช้ 6/16 ตาม LeNet-5
- p.128-130: คำอธิบาย FP/FN ของ churn สลับกัน; TN ต้องเป็น "Correctly"

## API ของ nnlab (อ่าน source ใน lab/src/nnlab/ ก่อนใช้ทุกครั้ง)
utils: set_seed, Timer, data_dir · conventions: to_deck, to_lib, assert_deck, describe · activations: sigmoid/tanh/relu/softmax, d_*, numerical_derivative, ACTIVATIONS
losses: bce_loss, binary_cross_entropy, cross_entropy, l2_penalty · preprocessing: one_hot, one_hot_dataframe, StandardNormalizer
data: load_lung_cancer_toy, load_churn_toy, load_churn_synthetic, load_breast_cancer_scaled, load_moons, load_blobs, load_digits_images, load_mnist, stratified_split, kfold_indices, get_dataset
metrics: confusion_matrix, binary_counts, accuracy/precision/recall/specificity/f1, binary_report, roc_curve, auc, per_class_counts, multiclass_report
conv: DECK_IMAGE_6x6, VERTICAL_EDGE, HORIZONTAL_EDGE, POOL_EXAMPLE_4x4, conv_output_size, zero_pad, conv2d_single, conv2d_volume, max_pool2d, avg_pool2d, describe_cnn, format_cnn_table
perceptron: forward, backward, naive_epoch, gradient_check, Perceptron · nn: init_params, forward, backward, compute_cost, gradient_check, iterate_minibatches, NeuralNetwork
optimizers: SGD, Momentum, RMSProp, Adam, lr_decay, make_optimizer · baselines: SklearnLogReg, SklearnMLP
torch_models: TorchPerceptron, TorchMLP, LeNet5, ResidualBlock, TorchTrainer, count_parameters · plotting: plot_history, plot_decision_boundary, plot_confusion, plot_roc, show_images, show_filters, show_feature_maps
config/experiment: TrainConfig, run, build_model, load_run, latest_run
vision (lab13): ImageCSVDataset, read_image_rgb, make_transforms, denormalize, cifar3_root, load_cifar3_loaders, SmallCNN, TransferResNet18, build_resnet18_transfer, set_trainable, freeze_backbone, unfreeze_layer4, trainable_parameters, describe_trainable, format_trainable_table, param_groups, extract_features · torch_models เพิ่ม: EarlyStopping, TorchTrainer(early_stopping_patience, checkpoint_path, param_groups, optimizer=prebuilt), predict_loader, evaluate_loader, count_parameters(trainable_only)


## แบบฝึกหัดแบบมี guide (เพิ่ม 2026-09-13 — แทนส่วน "ลองทำเอง" เดิม)
- ใช้ `nb.exercises_intro(lab_no)` → `nb.exercise(...)` 2-4 ข้อ → `nb.exercises_summary()` แทน `nb.exercises([...])`
- แต่ละข้อ: `ex_id` เช่น "5.1", `title`, `goal` (1 ประโยค), `steps` (2-5 ขั้น บอกว่าต้องเขียนอะไร ใช้ตัวแปร/ฟังก์ชันไหนจาก lab), `skeleton` (โค้ดโครง: signature ครบ, comment บอก shape, `raise NotImplementedError("ยังไม่ได้ทำ")` ตรงจุดที่ต้องเขียน; ถ้ามีหลายจุดใส่หลาย raise ได้), `check_code` (ใช้เฉพาะ `check`/`check_close`/`check_shape` จาก nnlab.exercise ห้าม assert/raise), `hints` (1-3 ข้อ พับได้)
- ระดับ: ข้อแรกง่าย (เติม 1-3 บรรทัด), ข้อกลางเขียนฟังก์ชันสั้น, ข้อสุดท้ายประยุกต์/ทดลอง (เช่น เทรนด้วย hyperparameter ที่กำหนดแล้วตรวจว่า accuracy ≥ เกณฑ์)
- check ต้อง "ไม่เฉลย": เทียบกับฟังก์ชันใน nnlab, ค่าคงที่ที่รู้ (จากสไลด์), หรือคุณสมบัติ (shape, ช่วงค่า, ผลรวม = 1, accuracy ≥ เกณฑ์) — ห้ามใส่โค้ดเฉลยใน check
- ใช้ตัวแปรที่มีอยู่แล้วใน notebook ได้ (เช่น X, y, rng) แต่ระบุใน steps ว่าใช้ตัวไหน
- **cell โครงต้องรันผ่านเสมอ (สำคัญที่สุด)**: `raise NotImplementedError` อยู่ได้เฉพาะ**ในตัวฟังก์ชัน**; ตัวแปรที่ให้เติมใช้ `x = ...` (Ellipsis → cell ตรวจรายงาน ⏳ เอง); ห้ามเรียกฟังก์ชันของนิสิตหรือใช้ตัวแปร `...` ทำคณิตศาสตร์ที่ระดับบนสุดของ cell โครง (ใส่ตัวอย่างการเรียกเป็น comment แทน) — ไม่งั้น nbconvert/pytest จะล้มทั้ง notebook
- ใน cell ตรวจ ถ้าต้องเรียกฟังก์ชันของนิสิตนอก check(...) ให้ห่อด้วย try หรือใส่ไว้ใน lambda เท่านั้น
- notebook ต้องรันผ่านทั้งไฟล์โดยที่ skeleton ยังไม่ได้ทำ → cell ตรวจพิมพ์ ⏳ (ห้าม error) — ตรวจด้วย `uv run python tools/exercise_check.py notebooks/labXX_*.ipynb` (ไม่ใส่เฉลย) ต้องได้ cell error 0
- เฉลยเก็บนอก repo: `<course>/lab-solutions/labXX_solutions.py` เป็น dict `SOLUTIONS = {"5.1": '''โค้ดเต็มของ cell โครง''', ...}` ตรวจด้วย `uv run python tools/exercise_check.py notebooks/labXX_*.ipynb ../lab-solutions/labXX_solutions.py` ต้อง PASS ทุกข้อ
