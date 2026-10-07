# Run the demo (Step 8)

## Prerequisites

- Trained checkpoint: `models/fusion_detector_best.pt`
- Python deps: `pip install -r requirements.txt`
- Node 18+

## 1. Start FastAPI (terminal 1)

From project root:

```bash
cd "/Users/admin/Documents/Intelligent Systems/Project 2"
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Check: http://127.0.0.1:8000/health

## 2. Start Next.js (terminal 2)

```bash
cd web
npm install
npm run dev
```

Open: http://localhost:3000

Upload an image → **Detect** → see REAL/FAKE, confidence, branch contributions.

## API

- `POST http://127.0.0.1:8000/predict` — multipart field `file` (image)
- Next.js proxies via `POST /api/predict`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `Checkpoint not found` | Train or copy `fusion_detector_best.pt` into `models/` |
| `Cannot reach ML API` | Start uvicorn first |
| Slow first request | Model loads on startup (~10–30s) |
| MPS/CUDA | Auto-detected; CPU works but slower |
