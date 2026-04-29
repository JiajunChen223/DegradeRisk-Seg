from __future__ import annotations

from typing import Any

import numpy as np

from src.descriptors.analysis_only.mask_stats import extract_mask_descriptors
from src.descriptors.input_safe.features import extract_input_safe_descriptors


def extract_descriptors(
    x: np.ndarray,
    y: np.ndarray | None,
    obs_mask: np.ndarray,
    feature_names: list[str],
    feature_families: list[str],
    feature_modalities: list[str],
    degradation_meta: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "input_safe": extract_input_safe_descriptors(
            x=x,
            obs_mask=obs_mask,
            feature_names=feature_names,
            feature_modalities=feature_modalities,
            degradation_meta=degradation_meta,
        )
    }
    if y is not None:
        payload["analysis_only"] = extract_mask_descriptors(y)
    return payload
