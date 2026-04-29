from __future__ import annotations

import numpy as np
import torch


def _prepare_binary(logits: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
    labels = target.detach().cpu().numpy().reshape(-1)
    return probs, labels


def _confidence_correctness(logits: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probs, labels = _prepare_binary(logits, target)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    confidence = np.maximum(probs, 1 - probs)
    correctness = ((probs >= 0.5).astype(np.float32) == labels.astype(np.float32)).astype(np.float32)
    return probs, confidence, correctness


def compute_binary_calibration_metrics(logits: torch.Tensor, target: torch.Tensor, bins: int = 15) -> dict[str, float]:
    probs, labels = _prepare_binary(logits, target)
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    _, conf, correct = _confidence_correctness(logits, target)
    nll = float(-(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean())
    brier = float(((probs - labels) ** 2).mean())
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for idx in range(bins):
        left, right = edges[idx], edges[idx + 1]
        mask = (conf >= left) & (conf < right if idx < bins - 1 else conf <= right)
        if not np.any(mask):
            continue
        acc = float(correct[mask].mean())
        avg_conf = float(conf[mask].mean())
        ece += (mask.mean()) * abs(acc - avg_conf)
    return {"ece": float(ece), "nll": nll, "brier": brier}


def calibration_bin_table(logits: torch.Tensor, target: torch.Tensor, bins: int = 15) -> list[dict[str, float | int]]:
    _, confidence, correctness = _confidence_correctness(logits, target)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float | int]] = []
    total = max(int(confidence.size), 1)
    for idx in range(bins):
        left, right = float(edges[idx]), float(edges[idx + 1])
        mask = (confidence >= left) & (confidence < right if idx < bins - 1 else confidence <= right)
        count = int(mask.sum())
        if count:
            accuracy = float(correctness[mask].mean())
            avg_confidence = float(confidence[mask].mean())
        else:
            accuracy = float("nan")
            avg_confidence = float("nan")
        signed_gap = avg_confidence - accuracy if count else float("nan")
        abs_gap = abs(signed_gap) if count else float("nan")
        mass = count / total
        rows.append(
            {
                "bin_index": idx,
                "bin_left": left,
                "bin_right": right,
                "count": count,
                "mass": float(mass),
                "accuracy": accuracy,
                "confidence": avg_confidence,
                "signed_gap": signed_gap,
                "abs_gap": abs_gap,
                "ece_contribution": float(mass * abs_gap) if count else 0.0,
                "overconfidence_contribution": float(mass * max(signed_gap, 0.0)) if count else 0.0,
                "underconfidence_contribution": float(mass * max(-signed_gap, 0.0)) if count else 0.0,
            }
        )
    return rows


def compute_calibration_decomposition(
    logits: torch.Tensor,
    target: torch.Tensor,
    bins: int = 15,
    high_confidence_threshold: float = 0.9,
) -> dict[str, float]:
    _, confidence, correctness = _confidence_correctness(logits, target)
    bin_rows = calibration_bin_table(logits, target, bins=bins)
    high_mask = confidence >= float(high_confidence_threshold)
    high_error = 1.0 - correctness[high_mask] if np.any(high_mask) else np.asarray([], dtype=np.float32)
    return {
        "ece": float(sum(float(row["ece_contribution"]) for row in bin_rows)),
        "overconfidence_ece": float(sum(float(row["overconfidence_contribution"]) for row in bin_rows)),
        "underconfidence_ece": float(sum(float(row["underconfidence_contribution"]) for row in bin_rows)),
        "mean_confidence": float(confidence.mean()),
        "mean_accuracy": float(correctness.mean()),
        "mean_signed_gap": float(confidence.mean() - correctness.mean()),
        "high_confidence_threshold": float(high_confidence_threshold),
        "high_confidence_mass": float(high_mask.mean()),
        "high_confidence_error_rate": float(high_error.mean()) if high_error.size else float("nan"),
    }
