"""Fusion detector """

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

SPATIAL_DIM = 1536
FREQ_DIM = 256
NOISE_DIM = 128

ABLATION_CONFIGS: dict[str, dict[str, bool]] = {
    "A": {"use_spatial": True, "use_frequency": False, "use_noise": False},
    "B": {"use_spatial": False, "use_frequency": True, "use_noise": False},
    "C": {"use_spatial": False, "use_frequency": False, "use_noise": True},
    "A+B": {"use_spatial": True, "use_frequency": True, "use_noise": False},
    "A+C": {"use_spatial": True, "use_frequency": False, "use_noise": True},
    "B+C": {"use_spatial": False, "use_frequency": True, "use_noise": True},
    "A+B+C": {"use_spatial": True, "use_frequency": True, "use_noise": True},
}


class SpatialBranch(nn.Module):
    out_dim: int = SPATIAL_DIM

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class FrequencyBranch(nn.Module):
    out_dim: int = FREQ_DIM

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        return self.proj(torch.flatten(x, 1))


class NoiseBranch(nn.Module):
    out_dim: int = NOISE_DIM

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(64, self.out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc(x)
        return self.proj(torch.flatten(x, 1))


class FusionHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionDetector(nn.Module):
    def __init__(
        self,
        use_spatial: bool = True,
        use_frequency: bool = True,
        use_noise: bool = True,
        pretrained_spatial: bool = False,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if not any([use_spatial, use_frequency, use_noise]):
            raise ValueError("At least one branch must be enabled.")
        self.use_spatial = use_spatial
        self.use_frequency = use_frequency
        self.use_noise = use_noise
        self.spatial = SpatialBranch(pretrained=pretrained_spatial) if use_spatial else None
        self.frequency = FrequencyBranch() if use_frequency else None
        self.noise = NoiseBranch() if use_noise else None
        fuse_dim = sum(
            d for flag, d in (
                (use_spatial, SPATIAL_DIM),
                (use_frequency, FREQ_DIM),
                (use_noise, NOISE_DIM),
            )
            if flag
        )
        self.fusion = FusionHead(fuse_dim, num_classes=num_classes)

    def extract_branch_features(
        self,
        spatial: torch.Tensor,
        frequency: torch.Tensor,
        noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        if self.spatial is not None:
            out["spatial"] = self.spatial(spatial)
        if self.frequency is not None:
            out["frequency"] = self.frequency(frequency)
        if self.noise is not None:
            out["noise"] = self.noise(noise)
        return out

    def forward(
        self,
        spatial: torch.Tensor,
        frequency: torch.Tensor,
        noise: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        branch_feats = self.extract_branch_features(spatial, frequency, noise)
        fused = torch.cat(list(branch_feats.values()), dim=1)
        return {"logits": self.fusion(fused), "branch_features": branch_feats}

    def forward_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        return self.forward(
            batch["spatial"].to(device),
            batch["frequency"].to(device),
            batch["noise"].to(device),
        )


def build_model(config_name: str = "A+B+C", pretrained_spatial: bool = False) -> FusionDetector:
    if config_name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown config {config_name!r}")
    return FusionDetector(pretrained_spatial=pretrained_spatial, **ABLATION_CONFIGS[config_name])
