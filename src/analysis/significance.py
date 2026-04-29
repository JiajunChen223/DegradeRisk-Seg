from __future__ import annotations

import numpy as np


def bootstrap_mean_ci(values: np.ndarray, rounds: int = 1000, confidence: float = 0.95, seed: int = 1) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if values.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    means = []
    for _ in range(rounds):
        sample = rng.choice(values, size=values.size, replace=True)
        means.append(float(sample.mean()))
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(values.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
    }
