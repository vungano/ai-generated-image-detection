"""Image preprocessing """

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from scipy.fft import dctn
from torchvision import transforms

import cv2

from api.config import Settings


def get_geometric_transforms(settings: Settings) -> transforms.Compose:
    size = settings.img_size
    bicubic = transforms.InterpolationMode.BICUBIC
    return transforms.Compose([
        transforms.Resize((size, size), interpolation=bicubic),
    ])


def get_spatial_head_transforms(settings: Settings) -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=settings.imagenet_mean, std=settings.imagenet_std),
    ])


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return ((arr - lo) / (hi - lo + 1e-8)).astype(np.float32)


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)


def compute_fft_map(gray: np.ndarray) -> np.ndarray:
    spectrum = np.fft.fft2(gray)
    shifted = np.fft.fftshift(spectrum)
    return _minmax_norm(np.log1p(np.abs(shifted)))


def compute_dct_map(gray: np.ndarray, block_size: int) -> np.ndarray:
    h, w = gray.shape
    bh, bw = h // block_size, w // block_size
    grid = np.zeros((bh, bw), dtype=np.float32)
    for bi in range(bh):
        for bj in range(bw):
            block = gray[bi * block_size : (bi + 1) * block_size, bj * block_size : (bj + 1) * block_size]
            coeffs = dctn(block, norm="ortho")
            grid[bi, bj] = np.mean(np.abs(coeffs[1:, 1:]))
    dct_full = np.repeat(np.repeat(grid, block_size, axis=0), block_size, axis=1)
    return _minmax_norm(dct_full[:h, :w])


def compute_noise_residual(rgb: np.ndarray, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(rgb, ksize=(0, 0), sigmaX=sigma)
    residual = rgb.astype(np.float32) - blurred.astype(np.float32)
    for c in range(3):
        ch = residual[..., c]
        residual[..., c] = ch / (np.max(np.abs(ch)) + 1e-8)
    return residual.astype(np.float32)


def build_frequency_tensor(rgb: np.ndarray, block_size: int) -> torch.Tensor:
    gray = rgb_to_gray(rgb)
    return torch.from_numpy(np.stack([compute_fft_map(gray), compute_dct_map(gray, block_size)], axis=0))


def build_noise_tensor(rgb: np.ndarray, sigma: float) -> torch.Tensor:
    residual = compute_noise_residual(rgb, sigma)
    return torch.from_numpy(residual.transpose(2, 0, 1).copy())


def pil_to_model_inputs(pil_img: Image.Image, settings: Settings) -> dict[str, torch.Tensor]:
    img = pil_img.convert("RGB")
    geom_img = get_geometric_transforms(settings)(img)
    spatial = get_spatial_head_transforms(settings)(geom_img)
    rgb = np.asarray(geom_img, dtype=np.float32) / 255.0
    return {
        "spatial": spatial.unsqueeze(0),
        "frequency": build_frequency_tensor(rgb, settings.dct_block).unsqueeze(0),
        "noise": build_noise_tensor(rgb, settings.noise_sigma).unsqueeze(0),
    }
