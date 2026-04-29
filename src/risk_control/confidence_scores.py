from __future__ import annotations

import numpy as np
import torch


def patch_confidence_scores(logits: torch.Tensor, score_name: str = "msp") -> np.ndarray:
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    conf = np.maximum(probs, 1.0 - probs)
    if score_name == "msp":
        return conf.mean(axis=(1, 2, 3))
    if score_name == "margin":
        return np.abs(probs - 0.5).mean(axis=(1, 2, 3)) * 2.0
    if score_name == "entropy":
        probs = np.clip(probs, 1e-6, 1 - 1e-6)
        ent = -(probs * np.log(probs) + (1 - probs) * np.log(1 - probs))
        return 1.0 - ent.mean(axis=(1, 2, 3)) / np.log(2.0)
    raise KeyError(f"Unknown confidence score: {score_name}")

