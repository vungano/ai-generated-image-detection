# Overnight full training — checklist (Distinction path)

## Do you need a better dataset?

**No — for Distinction, train on CIFAKE only.**

| Use CIFAKE for | Why |
|----------------|-----|
| **Training** | Matches your report, architecture, and rubric |
| **Test metrics** | Official 20k held-out split |
| **Demo / video** | Use `data/cifake/test/REAL` and `FAKE` only |

**Do not** claim the demo works on “any image.” In the video and report, say:

> *Demonstration uses in-distribution CIFAKE test images. Out-of-distribution images (other generators, full-resolution photos) are expected to fail—see Discussion.*

**Optional (1–2 hours, no retrain):** download **one** GenImage generator folder → run Step 6 with `RUN_GENIMAGE=True` → add a paragraph + small table to the report. That strengthens **Solution & Discussion (40%)** without changing overnight training.

---

## Before you start (5 minutes)

1. **Config cell** in the notebook:
   ```python
   auto_train = True
   force_retrain = True
   batch_size = 32        # use 16 if GPU runs out of memory
   ```

2. **First cell** must show:
   - `Device: mps` (Mac) or `cuda` (PC/NVIDIA)
   - `MPS compiled: True` or CUDA available

3. **Plug in power**, disable sleep, close heavy apps.

4. **Backup** current checkpoint (optional):
   ```bash
   cp models/fusion_detector_best.pt models/fusion_detector_epoch1_backup.pt
   ```

---

## What to run

**Option A — Notebook (simplest)**  
Kernel → **Restart & Run All** (leave it overnight).

- Skips retrain if checkpoint exists **unless** `force_retrain=True` ✓  
- Runs smoke test (~10 min) then full training (~3–8 h on GPU).

**Option B — Skip smoke** (save ~10 min)  
In §4.6, temporarily set at top of the training cell:
```python
RUN_SMOKE_ONLY = False  # add this flag if you add skip logic
```
Or comment out the smoke block manually—optional only.

---

## What “done” looks like in the morning

- `models/fusion_detector_best.pt` updated  
- `models/fusion_detector_best_history.json` shows **multiple epochs** and phases  
- Console ended with `Full training done. best val F1=...`

Then **only re-run**:
1. Step 5 (test evaluation)  
2. §4.6 load cell *or* restart with `auto_train=False`  
3. Optional: Step 6 robustness + Grad-CAM  
4. Update **Table 1** in `CSYM015_Report.md` from new `outputs/test_metrics.json`  
5. Change report line about “epoch 1 only” → “full three-phase training completed”

Restart FastAPI so it loads the new weights.

---

## Demo & video (Distinction)

| Do | Don't |
|----|--------|
| Upload 2 REAL + 2 FAKE from `data/cifake/test/` | Random Google images |
| Show metrics table in video | Claim universal detector |
| Show one OOD fail + explain why | Hide generalisation limit |

---

## Expected results after full training

Often **similar or slightly better** than epoch 1 (F1 ~0.81–0.88, recall still high). Biggest gain for markers: **“completed training protocol”** not necessarily +10% accuracy.

---

## If training stops early

- Check `fusion_detector_best.pt` anyway (best epoch saved)  
- Re-run Step 5 with that checkpoint  
- Report early stopping honestly in Methodology
