from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def risk_coverage_curve(scores: np.ndarray, losses: np.ndarray) -> dict[str, np.ndarray | float]:
    order = np.argsort(scores)[::-1]
    sorted_losses = losses[order]
    coverages = []
    risks = []
    for keep in range(1, len(order) + 1):
        retained = sorted_losses[:keep]
        coverages.append(keep / len(order))
        risks.append(float(retained.mean()))
    aurc = float(np.trapezoid(risks, coverages))
    return {"coverage": np.asarray(coverages), "risk": np.asarray(risks), "aurc": aurc}


def _threshold_for_coverage(scores: np.ndarray, target_coverage: float) -> float:
    quantile = max(0.0, min(1.0, 1.0 - target_coverage))
    return float(np.quantile(scores, quantile))


def select_global_thresholds(scores: np.ndarray, target_coverages: Iterable[float]) -> dict[str, float]:
    return {f"{coverage:.2f}": _threshold_for_coverage(scores, coverage) for coverage in target_coverages}


def select_groupwise_thresholds(scores: np.ndarray, groups: Iterable[str], target_coverages: Iterable[float]) -> dict[str, dict[str, float]]:
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    for score, group in zip(scores.tolist(), groups):
        grouped_scores[str(group)].append(score)
    payload: dict[str, dict[str, float]] = {}
    for group, values in grouped_scores.items():
        arr = np.asarray(values)
        payload[group] = select_global_thresholds(arr, target_coverages)
    return payload


def apply_global_threshold(scores: np.ndarray, threshold: float) -> np.ndarray:
    return scores >= threshold


def apply_groupwise_threshold(scores: np.ndarray, groups: Iterable[str], thresholds: dict[str, float], default: float | None = None) -> np.ndarray:
    if default is None:
        default = min(thresholds.values()) if thresholds else 0.0
    keep = []
    for score, group in zip(scores.tolist(), groups):
        threshold = thresholds.get(str(group), default)
        keep.append(score >= threshold)
    return np.asarray(keep, dtype=bool)


def summarize_retention(losses: np.ndarray, keep_mask: np.ndarray) -> dict[str, float]:
    coverage = float(keep_mask.mean())
    if coverage <= 0.0:
        return {"coverage": 0.0, "retained_risk": float("nan")}
    return {"coverage": coverage, "retained_risk": float(losses[keep_mask].mean())}


def summarize_group_retention(losses: np.ndarray, keep_mask: np.ndarray, groups: Iterable[str]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    groups_arr = np.asarray([str(group) for group in groups])
    for group in sorted(set(groups_arr.tolist())):
        group_mask = groups_arr == group
        group_keep = keep_mask[group_mask]
        group_losses = losses[group_mask]
        retained = group_losses[group_keep]
        rows.append(
            {
                "group_label": group,
                "group_count": int(group_mask.sum()),
                "group_coverage": float(group_keep.mean()) if group_keep.size else 0.0,
                "group_retained_count": int(group_keep.sum()),
                "group_retained_risk": float(retained.mean()) if retained.size else float("nan"),
            }
        )
    return rows


def worst_group_retention(group_rows: list[dict[str, float | str | int]]) -> dict[str, float | str | int]:
    valid_rows = [row for row in group_rows if not np.isnan(float(row["group_retained_risk"]))]
    if not valid_rows:
        return {
            "worst_group_label": "na",
            "worst_group_retained_risk": float("nan"),
            "worst_group_count": 0,
            "worst_group_risk_gap": float("nan"),
        }
    ordered = sorted(valid_rows, key=lambda row: (float(row["group_retained_risk"]), int(row["group_count"])), reverse=True)
    worst = ordered[0]
    risks = np.asarray([float(row["group_retained_risk"]) for row in valid_rows], dtype=np.float32)
    return {
        "worst_group_label": str(worst["group_label"]),
        "worst_group_retained_risk": float(worst["group_retained_risk"]),
        "worst_group_count": int(worst["group_count"]),
        "worst_group_risk_gap": float(risks.max() - risks.min()) if risks.size else float("nan"),
    }
