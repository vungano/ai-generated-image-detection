"""Load checkpoint and run inference."""

from __future__ import annotations

import io
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from api.config import Settings, load_settings
from api.model import build_model
from api.preprocess import pil_to_model_inputs


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def branch_contributions(branch_feats: dict[str, torch.Tensor]) -> dict[str, float]:
    """Relative L2 norm per branch (sums to 1)."""
    keys = {"spatial": "spatial", "frequency": "frequency", "noise": "noise_residual"}
    norms = {}
    for k, v in branch_feats.items():
        norms[keys.get(k, k)] = float(v.norm(dim=1).item())
    total = sum(norms.values()) or 1.0
    return {k: round(v / total, 4) for k, v in norms.items()}


class Predictor:
    def __init__(self, checkpoint_path: Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self.device = get_device()
        path = checkpoint_path or self.settings.checkpoint_path
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        ablation = ckpt.get("ablation", "A+B+C")
        self.model = build_model(ablation, pretrained_spatial=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()
        self.meta = ckpt

    @torch.no_grad()
    def predict_image(self, pil_img: Image.Image) -> dict:
        t0 = time.perf_counter()
        batch = pil_to_model_inputs(pil_img, self.settings)
        out = self.model.forward_batch(batch)
        probs = F.softmax(out["logits"], dim=1)[0]
        pred_idx = int(probs.argmax().item())
        label = self.settings.class_names[pred_idx]
        conf = float(probs[pred_idx].item())
        branches = branch_contributions(out["branch_features"])
        ms = int((time.perf_counter() - t0) * 1000)
        return {
            "prediction": label,
            "confidence": round(conf, 4),
            "prob_fake": round(float(probs[1].item()), 4),
            "branch_contributions": branches,
            "processing_time_ms": ms,
        }

    def predict_bytes(self, data: bytes) -> dict:
        return self.predict_image(Image.open(io.BytesIO(data)).convert("RGB"))
