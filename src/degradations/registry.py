from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np


def _stable_rng(seed: int, sample_key: str, family_name: str, severity_name: str) -> np.random.Generator:
    digest = hashlib.md5(f"{seed}-{sample_key}-{family_name}-{severity_name}".encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:8], 16))


def _month_level_gap(obs_mask: np.ndarray) -> int:
    month_valid = obs_mask.max(axis=1) > 0
    best = 0
    running = 0
    for flag in month_valid.tolist():
        if flag:
            running = 0
        else:
            running += 1
            best = max(best, running)
    return best


def _make_meta(
    family_name: str,
    severity_name: str,
    obs_mask: np.ndarray,
    feature_families: list[str],
    feature_modalities: list[str],
) -> dict[str, Any]:
    modality_arr = np.asarray(feature_modalities)
    missing_sar = float(obs_mask[:, modality_arr == "SAR"].mean()) if np.any(modality_arr == "SAR") else 1.0
    missing_opt = float(obs_mask[:, modality_arr == "OPT"].mean()) if np.any(modality_arr == "OPT") else 1.0
    return {
        "family": family_name,
        "severity": severity_name,
        "retained_budget": float(obs_mask.mean()),
        "longest_gap": int(_month_level_gap(obs_mask)),
        "sar_retained": missing_sar,
        "opt_retained": missing_opt,
        "feature_family_count": len(dict.fromkeys(feature_families)),
        "feature_family_labels": list(dict.fromkeys(feature_families)),
        "feature_modality_labels": list(dict.fromkeys(feature_modalities)),
    }


def _apply_random_missing_months(x: np.ndarray, obs_mask: np.ndarray, rng: np.random.Generator, params: dict[str, Any]) -> None:
    drop_k = int(params["drop_k"])
    months = rng.choice(x.shape[0], size=min(drop_k, x.shape[0]), replace=False)
    obs_mask[months, :] = 0.0
    x[months] = 0.0


def _apply_block_missing_months(x: np.ndarray, obs_mask: np.ndarray, rng: np.random.Generator, params: dict[str, Any]) -> None:
    block_len = int(params["block_len"])
    block_len = min(block_len, x.shape[0])
    start = int(rng.integers(0, x.shape[0] - block_len + 1))
    obs_mask[start : start + block_len, :] = 0.0
    x[start : start + block_len] = 0.0


def _apply_prefix_truncation(x: np.ndarray, obs_mask: np.ndarray, params: dict[str, Any]) -> None:
    keep_t = int(params["keep_t"])
    keep_t = min(max(keep_t, 0), x.shape[0])
    obs_mask[keep_t:, :] = 0.0
    x[keep_t:] = 0.0


def _apply_modality_drop(x: np.ndarray, obs_mask: np.ndarray, feature_modalities: list[str], params: dict[str, Any]) -> None:
    drop_group = str(params["drop_group"]).upper()
    channel_ids = [idx for idx, modality in enumerate(feature_modalities) if modality.upper() == drop_group]
    if not channel_ids:
        return
    obs_mask[:, channel_ids] = 0.0
    x[:, channel_ids] = 0.0


def _apply_irregular_missing(
    x: np.ndarray,
    obs_mask: np.ndarray,
    feature_families: list[str],
    rng: np.random.Generator,
    params: dict[str, Any],
) -> None:
    retain_rate = float(params["retain_rate"])
    unique_families = sorted(set(feature_families))
    for month in range(x.shape[0]):
        for family in unique_families:
            keep = float(rng.random()) <= retain_rate
            if keep:
                continue
            channel_ids = [idx for idx, item in enumerate(feature_families) if item == family]
            obs_mask[month, channel_ids] = 0.0
            x[month, channel_ids] = 0.0


def _channel_scale(x: np.ndarray, channel_ids: list[int]) -> np.ndarray:
    values = x[:, channel_ids]
    scale = values.std(axis=(0, 2, 3), keepdims=True)
    return np.clip(scale, 1.0e-6, None)


def _apply_modality_aware_radiometric_perturbation(
    x: np.ndarray,
    feature_modalities: list[str],
    rng: np.random.Generator,
    params: dict[str, Any],
) -> None:
    modality_arr = np.asarray([item.upper() for item in feature_modalities])
    sar_ids = np.flatnonzero(modality_arr == "SAR").tolist()
    opt_ids = np.flatnonzero(modality_arr == "OPT").tolist()
    if sar_ids:
        sar_sigma = float(params.get("sar_multiplicative_std", 0.0))
        sar_additive_std = float(params.get("sar_additive_std", 0.0))
        sar_shape = (x.shape[0], len(sar_ids), 1, 1)
        speckle = rng.lognormal(mean=-0.5 * sar_sigma**2, sigma=sar_sigma, size=sar_shape).astype(np.float32)
        x[:, sar_ids] *= speckle
        if sar_additive_std > 0.0:
            scale = _channel_scale(x, sar_ids)
            x[:, sar_ids] += rng.normal(0.0, sar_additive_std, size=x[:, sar_ids].shape).astype(np.float32) * scale
    if opt_ids:
        opt_gain_std = float(params.get("opt_gain_std", 0.0))
        opt_additive_std = float(params.get("opt_additive_std", 0.0))
        opt_shape = (x.shape[0], len(opt_ids), 1, 1)
        gain = (1.0 + rng.normal(0.0, opt_gain_std, size=opt_shape)).astype(np.float32)
        x[:, opt_ids] *= gain
        if opt_additive_std > 0.0:
            scale = _channel_scale(x, opt_ids)
            x[:, opt_ids] += rng.normal(0.0, opt_additive_std, size=x[:, opt_ids].shape).astype(np.float32) * scale
    outlier_prob = float(params.get("outlier_prob", 0.0))
    if outlier_prob <= 0.0:
        return
    outlier_mask = rng.random(x.shape) <= outlier_prob
    all_ids = list(range(x.shape[1]))
    scale = _channel_scale(x, all_ids)
    outlier_noise = rng.normal(0.0, 4.0, size=x.shape).astype(np.float32) * scale
    x += outlier_mask.astype(np.float32) * outlier_noise


def build_degrader(config: dict[str, Any], family_name: str | None = None, severity_name: str | None = None) -> Callable:
    degradation_cfg = config.get("degradation", {})
    if not degradation_cfg.get("enabled", False) or family_name == "clean":
        return lambda x, feature_families, feature_modalities, seed, sample_key: (
            x,
            np.ones((x.shape[0], x.shape[1]), dtype=np.float32),
            _make_meta("clean", "L0", np.ones((x.shape[0], x.shape[1]), dtype=np.float32), feature_families, feature_modalities),
        )

    family_name = family_name or degradation_cfg.get("default_eval_family", "random_missing_months")
    severity_name = severity_name or degradation_cfg.get("severity", "L2")
    family_cfg = degradation_cfg["families"][family_name]
    params = family_cfg["severities"][severity_name]

    def degrader(
        x: np.ndarray,
        feature_families: list[str],
        feature_modalities: list[str],
        seed: int,
        sample_key: str,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        x_deg = x.copy()
        obs_mask = np.ones((x_deg.shape[0], x_deg.shape[1]), dtype=np.float32)
        rng = _stable_rng(seed, sample_key, family_name, severity_name)
        if family_cfg["type"] == "random_missing_months":
            _apply_random_missing_months(x_deg, obs_mask, rng, params)
        elif family_cfg["type"] == "block_missing_months":
            _apply_block_missing_months(x_deg, obs_mask, rng, params)
        elif family_cfg["type"] == "prefix_truncation":
            _apply_prefix_truncation(x_deg, obs_mask, params)
        elif family_cfg["type"] == "modality_drop":
            _apply_modality_drop(x_deg, obs_mask, feature_modalities, params)
        elif family_cfg["type"] == "irregular_missing":
            _apply_irregular_missing(x_deg, obs_mask, feature_families, rng, params)
        elif family_cfg["type"] == "modality_aware_radiometric_perturbation":
            _apply_modality_aware_radiometric_perturbation(x_deg, feature_modalities, rng, params)
        else:
            raise KeyError(f"Unknown degradation family type: {family_cfg['type']}")
        return x_deg, obs_mask, _make_meta(family_name, severity_name, obs_mask, feature_families, feature_modalities)

    return degrader
