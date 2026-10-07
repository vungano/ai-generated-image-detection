"""Runtime config — mirrors notebook Config / outputs/config.json."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = PROJECT_ROOT / "models" / "fusion_detector_best.pt"
CONFIG_JSON = PROJECT_ROOT / "outputs" / "config.json"


@dataclass
class Settings:
    project_root: Path = PROJECT_ROOT
    checkpoint_path: Path = DEFAULT_CHECKPOINT
    img_size: int = 224
    imagenet_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    class_names: tuple[str, str] = ("REAL", "FAKE")
    noise_sigma: float = 1.5
    dct_block: int = 8


def load_settings() -> Settings:
    s = Settings()
    if CONFIG_JSON.exists():
        data = json.loads(CONFIG_JSON.read_text())
        if "img_size" in data:
            s.img_size = int(data["img_size"])
        if "class_names" in data:
            s.class_names = tuple(data["class_names"])
    return s
