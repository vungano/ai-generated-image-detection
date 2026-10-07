#!/usr/bin/env python3
"""Logistic regression baseline on handcrafted forensic features (full CIFAKE by default)."""
from __future__ import annotations
import json, random, time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from scipy.fft import dctn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# None = use all images in train/test folders (fair comparison with fusion model)
MAX_PER_CLASS_TRAIN = None
MAX_PER_CLASS_TEST = None


def list_images(root: Path, label: int, max_n: int | None = None):
    paths = sorted(p for p in root.iterdir() if p.suffix.lower() in IMG_EXTS)
    if max_n is not None:
        random.shuffle(paths)
        paths = paths[:max_n]
    return [(p, label) for p in paths]


def handcrafted_features(rgb: Image.Image) -> np.ndarray:
    arr = np.array(rgb.resize((224, 224), Image.BICUBIC), np.float32) / 255.0
    gray = cv2.cvtColor((arr * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    fft = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    h, w = fft.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    radial = [fft[r == ri].mean() for ri in range(1, 9)]
    blur = cv2.GaussianBlur(arr, (0, 0), 1.5)
    resid = arr - blur
    dct_e = dctn(gray, axes=(0, 1))
    dct_ac = float(np.abs(dct_e[1:9, 1:9]).mean())
    return np.array(
        list(arr.mean(axis=(0, 1))) + list(arr.std(axis=(0, 1))) +
        list(resid.std(axis=(0, 1))) + radial + [dct_ac, float(gray.std()), float(fft.max())],
        dtype=np.float32,
    )


def load_xy(root: Path, max_per_class: int | None):
    items = list_images(root / "REAL", 0, max_per_class) + list_images(root / "FAKE", 1, max_per_class)
    random.shuffle(items)
    X, y = [], []
    for path, label in tqdm(items, desc=root.name):
        X.append(handcrafted_features(Image.open(path).convert("RGB")))
        y.append(label)
    return np.stack(X), np.array(y)


def main():
    print("Extracting features (full dataset unless MAX_PER_CLASS_* set in script)...")
    t0 = time.time()
    X_tr, y_tr = load_xy(ROOT / "data/cifake/train", MAX_PER_CLASS_TRAIN)
    X_te, y_te = load_xy(ROOT / "data/cifake/test", MAX_PER_CLASS_TEST)
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)),
    ])
    clf.fit(X_tr, y_tr)
    prob = clf.predict_proba(X_te)[:, 1]
    pred = (prob >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", pos_label=1, zero_division=0)
    out = {
        "model": "Logistic Regression (24 handcrafted forensic features)",
        "train_n": int(len(y_tr)),
        "test_n": int(len(y_te)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc": float(roc_auc_score(y_te, prob)),
        "train_seconds": round(time.time() - t0, 1),
        "note": "Full CIFAKE train/test unless MAX_PER_CLASS_* capped in script",
    }
    print(json.dumps(out, indent=2))
    out_path = ROOT / "outputs/baseline_results.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
