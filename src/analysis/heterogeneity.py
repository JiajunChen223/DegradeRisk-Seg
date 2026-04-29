from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.analysis.significance import bootstrap_mean_ci


def fit_temporal_clusterer(curves: list[list[float]], n_clusters: int = 4, seed: int = 1) -> KMeans:
    if not curves:
        raise ValueError("No curves provided for temporal clustering")
    arr = np.asarray(curves, dtype=np.float32)
    model = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    model.fit(arr)
    return model


def assign_temporal_cluster(model: KMeans, curve: list[float]) -> int:
    arr = np.asarray(curve, dtype=np.float32).reshape(1, -1)
    return int(model.predict(arr)[0])


def fit_analysis_bin_edges(
    descriptors: list[dict[str, Any]],
    quantiles: tuple[float, float] = (0.33, 0.66),
) -> dict[str, tuple[float, float]]:
    analysis_keys = {
        "coverage_bin": "coverage",
        "boundary_bin": "boundary_complexity",
        "fragmentation_bin": "small_component_ratio",
    }
    edges: dict[str, tuple[float, float]] = {}
    for target_key, source_key in analysis_keys.items():
        values = [
            float(item.get("analysis_only", {}).get(source_key, 0.0))
            for item in descriptors
            if "analysis_only" in item
        ]
        if not values:
            edges[target_key] = (0.33, 0.66)
            continue
        arr = np.asarray(values, dtype=np.float32)
        low = float(np.quantile(arr, quantiles[0]))
        high = float(np.quantile(arr, quantiles[1]))
        if high < low:
            low, high = high, low
        edges[target_key] = (low, high)
    return edges


def _bin_from_edges(value: float, edges: tuple[float, float]) -> str:
    low, high = edges
    if value <= low:
        return "low"
    if value <= high:
        return "mid"
    return "high"


def _stable_group_seed(seed: int, *parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.md5(f"{seed}|{text}".encode("utf-8")).hexdigest()
    return seed + int(digest[:8], 16)


def build_group_record(
    sample_id: str,
    descriptors: dict[str, Any],
    metrics: dict[str, float],
    degradation_meta: dict[str, Any],
    temporal_cluster: int | None = None,
    analysis_edges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    input_safe = descriptors["input_safe"]
    analysis_only = descriptors.get("analysis_only", {})
    edges = analysis_edges or {
        "coverage_bin": (0.33, 0.66),
        "boundary_bin": (0.33, 0.66),
        "fragmentation_bin": (0.33, 0.66),
    }
    coverage = float(analysis_only.get("coverage", 0.0))
    boundary = float(analysis_only.get("boundary_complexity", 0.0))
    fragmentation = float(analysis_only.get("small_component_ratio", 0.0))
    record = {
        "sample_id": sample_id,
        "family": degradation_meta.get("family", "clean"),
        "severity": degradation_meta.get("severity", "L0"),
        "temporal_cluster": str(temporal_cluster if temporal_cluster is not None else -1),
        "budget_bin": input_safe["budget_bin"],
        "longest_gap_bin": input_safe["longest_gap_bin"],
        "roughness_bin": input_safe["roughness_bin"],
        "dominance_bin": input_safe["dominance_bin"],
        "coverage_bin": _bin_from_edges(coverage, edges["coverage_bin"]),
        "boundary_bin": _bin_from_edges(boundary, edges["boundary_bin"]),
        "fragmentation_bin": _bin_from_edges(fragmentation, edges["fragmentation_bin"]),
    }
    record.update(metrics)
    return record


def summarize_group_results(
    records: list[dict[str, Any]],
    rounds: int = 1000,
    confidence: float = 0.95,
    seed: int = 1,
    strata_keys: list[str] | None = None,
) -> pd.DataFrame:
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    strata = strata_keys or [
        "temporal_cluster",
        "budget_bin",
        "roughness_bin",
        "dominance_bin",
        "coverage_bin",
        "boundary_bin",
        "fragmentation_bin",
    ]
    metric_names = ["retained_risk", "ece", "dice"]
    rows: list[dict[str, Any]] = []
    for (family, severity), subset in frame.groupby(["family", "severity"]):
        for stratum in strata:
            if stratum not in subset.columns:
                continue
            for group_label, group_frame in subset.groupby(stratum):
                row: dict[str, Any] = {
                    "family": family,
                    "severity": severity,
                    "stratum": stratum,
                    "group_label": str(group_label),
                    "group_count": int(len(group_frame)),
                }
                for metric_name in metric_names:
                    ci = bootstrap_mean_ci(
                        group_frame[metric_name].to_numpy(dtype=np.float32),
                        rounds=rounds,
                        confidence=confidence,
                        seed=_stable_group_seed(seed, family, severity, stratum, group_label, metric_name),
                    )
                    row[metric_name] = ci["mean"]
                    row[f"{metric_name}_ci_low"] = ci["ci_low"]
                    row[f"{metric_name}_ci_high"] = ci["ci_high"]
                rows.append(row)
    return pd.DataFrame(rows)


def worst_group_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary
    rows: list[dict[str, Any]] = []
    for (family, severity), subset in summary.groupby(["family", "severity"]):
        risk_row = subset.sort_values(["retained_risk", "group_count"], ascending=[False, False]).iloc[0]
        ece_row = subset.sort_values(["ece", "group_count"], ascending=[False, False]).iloc[0]
        dice_row = subset.sort_values(["dice", "group_count"], ascending=[True, False]).iloc[0]
        rows.append(
            {
                "family": family,
                "severity": severity,
                "worst_group_risk": float(risk_row["retained_risk"]),
                "worst_group_risk_stratum": risk_row["stratum"],
                "worst_group_risk_label": risk_row["group_label"],
                "worst_group_risk_count": int(risk_row["group_count"]),
                "worst_group_risk_ci_low": float(risk_row["retained_risk_ci_low"]),
                "worst_group_risk_ci_high": float(risk_row["retained_risk_ci_high"]),
                "worst_group_ece": float(ece_row["ece"]),
                "worst_group_ece_stratum": ece_row["stratum"],
                "worst_group_ece_label": ece_row["group_label"],
                "worst_group_ece_count": int(ece_row["group_count"]),
                "worst_group_ece_ci_low": float(ece_row["ece_ci_low"]),
                "worst_group_ece_ci_high": float(ece_row["ece_ci_high"]),
                "worst_group_dice": float(dice_row["dice"]),
                "worst_group_dice_stratum": dice_row["stratum"],
                "worst_group_dice_label": dice_row["group_label"],
                "worst_group_dice_count": int(dice_row["group_count"]),
                "worst_group_dice_ci_low": float(dice_row["dice_ci_low"]),
                "worst_group_dice_ci_high": float(dice_row["dice_ci_high"]),
            }
        )
    return pd.DataFrame(rows)
