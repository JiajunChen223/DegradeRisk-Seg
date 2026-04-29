from __future__ import annotations

import numpy as np
import torch


def flatten_time_channels(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 5:
        raise ValueError(f"Expected [B, T, C, H, W], got {tuple(x.shape)}")
    b, t, c, h, w = x.shape
    return x.reshape(b, t * c, h, w)


def sigmoid_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def batch_dice_iou(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> dict[str, float]:
    probs = sigmoid_probs(logits)
    pred = (probs >= threshold).float()
    target = target.float().unsqueeze(1) if target.ndim == 3 else target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_area = pred.sum(dim=(1, 2, 3))
    target_area = target.sum(dim=(1, 2, 3))
    union = pred_area + target_area - intersection
    dice = ((2 * intersection + 1e-6) / (pred_area + target_area + 1e-6)).mean().item()
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    return {"dice": float(dice), "iou": float(iou)}


def per_sample_dice_loss(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    probs = sigmoid_probs(logits)
    pred = (probs >= threshold).float()
    target = target.float().unsqueeze(1) if target.ndim == 3 else target.float()
    intersection = (pred * target).sum(dim=(1, 2, 3))
    pred_area = pred.sum(dim=(1, 2, 3))
    target_area = target.sum(dim=(1, 2, 3))
    dice = (2 * intersection + 1e-6) / (pred_area + target_area + 1e-6)
    return (1.0 - dice).detach().cpu().numpy()

