from __future__ import annotations

from typing import Any

import numpy as np


def _bin_value(value: float, thresholds: tuple[float, float]) -> str:
    low, high = thresholds
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def extract_input_safe_descriptors(
    x: np.ndarray,
    obs_mask: np.ndarray,
    feature_names: list[str],
    feature_modalities: list[str],
    degradation_meta: dict[str, Any],
) -> dict[str, Any]:
    monthly_mean = x.mean(axis=(2, 3))
    monthly_presence = obs_mask.max(axis=1)
    modality_arr = np.asarray(feature_modalities)
    sar_ids = np.where(modality_arr == "SAR")[0]
    opt_ids = np.where(modality_arr == "OPT")[0]
    sar_curve = monthly_mean[:, sar_ids].mean(axis=1) if len(sar_ids) else np.zeros(x.shape[0], dtype=np.float32)
    opt_curve = monthly_mean[:, opt_ids].mean(axis=1) if len(opt_ids) else np.zeros(x.shape[0], dtype=np.float32)
    all_curve = monthly_mean.mean(axis=1)
    roughness = float(np.mean(np.abs(np.diff(all_curve)))) if len(all_curve) > 1 else 0.0
    anomaly_ratio = float(np.mean(np.abs(x - x.mean()) > (3.0 * max(x.std(), 1e-6))))
    sar_amp = float(sar_curve.max() - sar_curve.min()) if len(sar_ids) else 0.0
    opt_amp = float(opt_curve.max() - opt_curve.min()) if len(opt_ids) else 0.0
    dominance = float((sar_amp + 1e-6) / (opt_amp + 1e-6))
    phenology_shape_amplitude = float(all_curve.max() - all_curve.min()) if len(all_curve) else 0.0
    cross_modal_gap = float(abs(sar_amp - opt_amp))
    cross_modal_consistency = float(1.0 - min(cross_modal_gap / max(sar_amp + opt_amp, 1e-6), 1.0))
    retained_budget = float(degradation_meta.get("retained_budget", obs_mask.mean()))
    longest_gap = int(degradation_meta.get("longest_gap", 0))
    sar_retained = float(degradation_meta.get("sar_retained", 1.0))
    opt_retained = float(degradation_meta.get("opt_retained", 1.0))
    modality_flag = "multimodal"
    if sar_retained <= 0.0 < opt_retained:
        modality_flag = "opt_only"
    elif opt_retained <= 0.0 < sar_retained:
        modality_flag = "sar_only"
    elif sar_retained <= 0.0 and opt_retained <= 0.0:
        modality_flag = "no_signal"
    descriptors = {
        "descriptor_schema_version": "input_safe_v3",
        "retained_budget": retained_budget,
        "longest_gap": longest_gap,
        "sar_retained": sar_retained,
        "opt_retained": opt_retained,
        "roughness": roughness,
        "anomaly_ratio": anomaly_ratio,
        "dominance_ratio": dominance,
        "phenology_shape_amplitude": phenology_shape_amplitude,
        "cross_modal_consistency": cross_modal_consistency,
        "modality_flag": modality_flag,
        "temporal_curve": all_curve.tolist(),
        "monthly_presence": monthly_presence.astype(float).tolist(),
    }
    descriptors["budget_bin"] = _bin_value(retained_budget, (0.6, 0.85))
    descriptors["longest_gap_bin"] = "short" if longest_gap <= 1 else ("medium" if longest_gap <= 3 else "long")
    descriptors["roughness_bin"] = _bin_value(roughness, (0.08, 0.16))
    descriptors["dominance_bin"] = _bin_value(dominance, (0.8, 1.2))
    descriptors["phenology_shape_bin"] = _bin_value(phenology_shape_amplitude, (0.25, 0.50))
    descriptors["cross_modal_consistency_bin"] = _bin_value(cross_modal_consistency, (0.33, 0.66))
    descriptors["safe_group_key"] = "|".join(
        [
            descriptors["budget_bin"],
            descriptors["longest_gap_bin"],
            descriptors["phenology_shape_bin"],
            descriptors["cross_modal_consistency_bin"],
        ]
    )
    return descriptors
