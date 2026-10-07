# AI-Generated Image Detection

**CSYM015 — Intelligent Systems** · University of Northampton  
Student ID: **25818066** · Emmanuel Tinevimbo Vungano

Dual-branch frequency–spatial fusion detector with noise residual analysis. Distinguishes real photographs from AI-generated images using:

- **Branch A** — EfficientNet-B3 spatial features  
- **Branch B** — FFT + block DCT frequency maps  
- **Branch C** — PRNU-style noise residuals  
- **Fusion MLP** + asymmetric focal loss (recall-first on FAKE)

Primary benchmark: [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) (held-out test set, n = 20,000).

| Metric | Value |
|--------|------:|
| Accuracy | 98.6% |
| Precision (FAKE) | 97.9% |
| Recall (FAKE) | 99.3% |
| F1 | 0.986 |
| AUC-ROC | 0.999 |

> These scores are **in-distribution on CIFAKE**. Performance does not generalise reliably to arbitrary generators or full-resolution photos.

---

## Repository layout

```
├── api/                      # FastAPI inference service
├── web/                      # Next.js demo UI
├── scripts/                  # Baseline / helper scripts
├── models/                   # Place fusion_detector_best.pt here after training
├── data/cifake/              # Download CIFAKE here (not in git)
├── outputs/                  # Metrics JSON/CSV and evaluation plots
├── ai_image_detection.ipynb  # Training, EDA, ablation, robustness
├── requirements.txt
├── RUN_DEMO.md
└── CSYM015_Report.md         # Project report (markdown draft)
```

---

## Requirements

- Python 3.10+
- Node.js 18+
- Apple MPS, NVIDIA CUDA, or CPU

---

## Setup

### 1. Clone and install Python deps

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

On Apple Silicon, prefer pip-installed PyTorch for MPS:

```bash
pip install --upgrade torch torchvision
```

### 2. Download CIFAKE

1. Get the dataset from [Kaggle — CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).
2. Extract so paths look like:

```
data/cifake/train/{REAL,FAKE}/
data/cifake/test/{REAL,FAKE}/
```

### 3. Train (or provide a checkpoint)

Open and run the notebook:

```bash
jupyter notebook ai_image_detection.ipynb
```

This writes `models/fusion_detector_best.pt` (best validation F1).  
Training is not required if you already have that checkpoint locally.

---

## Run the demo

See [RUN_DEMO.md](RUN_DEMO.md). Short version:

**Terminal 1 — API**

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Web UI**

```bash
cd web
npm install
npm run dev
```

Open http://localhost:3000 — upload an image to get REAL / FAKE, confidence, and latency.

API: `POST http://127.0.0.1:8000/predict` (multipart field `file`).

---

## What is not in this repo

| Excluded | Why |
|----------|-----|
| `data/cifake/` | Large public dataset — download from Kaggle |
| `models/*.pt` | Checkpoint ~46 MB — train via the notebook |
| `web/node_modules/`, `web/.next/` | Rebuild with `npm install` / `npm run dev` |
| Report PDF/DOCX | Markdown report is included |

---

## Licence / coursework note

Submitted for CSYM015 coursework. Reuse for learning is fine; do not submit as your own assessed work.
