from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_protocol_matrix(output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    protocols = ["Random", "Block", "Prefix", "Modality", "Irregular", "Radiometric"]
    ax.imshow(np.arange(len(protocols)).reshape(1, -1), cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(protocols)), protocols, rotation=20)
    ax.set_yticks([0], ["Protocol"])
    ax.set_title("Fig.2 Budget-controlled degradation protocols")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _severity_order(value: str) -> int:
    order = {"L1": 1, "L2": 2, "L3": 3}
    return order.get(str(value), 99)


def plot_severity_curves(
    frame: pd.DataFrame,
    output_path: str | Path,
    metric: str = "dice",
    exclude_families: set[str] | None = None,
    title: str | None = None,
) -> None:
    filtered = frame.copy()
    excluded = set(exclude_families or {"modality_drop"})
    if "family" in filtered.columns and excluded:
        filtered = filtered.loc[~filtered["family"].isin(excluded)].copy()
    if filtered.empty or metric not in filtered.columns:
        return
    filtered["severity_order"] = filtered["severity"].map(_severity_order)
    fig, ax = plt.subplots(figsize=(7, 4))
    for family, sub in filtered.groupby("family"):
        ordered = sub.sort_values("severity_order")
        ax.plot(ordered["severity"], ordered[metric], marker="o", label=family)
    ax.set_xlabel("Severity")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Fig.3 {metric.upper()}-severity curves")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_modality_drop_summary(frame: pd.DataFrame, output_path: str | Path, metric: str = "dice") -> None:
    filtered = frame.loc[frame["family"] == "modality_drop"].copy()
    if filtered.empty or metric not in filtered.columns:
        return
    label_map = {"L1": "OPT drop", "L2": "SAR drop"}
    filtered = filtered.loc[filtered["severity"].isin(label_map)].copy()
    if filtered.empty:
        return
    filtered["label"] = filtered["severity"].map(label_map)
    filtered = filtered.sort_values("severity", key=lambda col: col.map(_severity_order))
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(filtered["label"], filtered[metric], color=["#5B8FF9", "#F6BD16"][: len(filtered)])
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Fig.3 {metric.upper()} under modality drop (non-ordinal)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_reliability_diagram(confidence: np.ndarray, correctness: np.ndarray, output_path: str | Path) -> None:
    bins = np.linspace(0.0, 1.0, 11)
    xs = []
    ys = []
    for idx in range(len(bins) - 1):
        mask = (confidence >= bins[idx]) & (confidence < bins[idx + 1] if idx < len(bins) - 2 else confidence <= bins[idx + 1])
        if not np.any(mask):
            continue
        xs.append(float(confidence[mask].mean()))
        ys.append(float(correctness[mask].mean()))
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.plot(xs, ys, marker="o")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig.4 Reliability diagram")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_risk_coverage(coverage: np.ndarray, risk: np.ndarray, output_path: str | Path, title: str = "Fig.5 Risk-coverage") -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(coverage, risk, marker="o")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_risk_coverage_comparison(curves: dict[str, tuple[np.ndarray, np.ndarray]], output_path: str | Path, title: str = "Fig.5 Risk-coverage") -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for label, (coverage, risk) in curves.items():
        ax.plot(coverage, risk, marker="o", label=label)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk")
    ax.set_title(title)
    if len(curves) > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_heterogeneity_heatmap(frame: pd.DataFrame, output_path: str | Path, value: str = "retained_risk") -> None:
    pivot = frame.pivot_table(index="family", columns="temporal_cluster", values=value, aggfunc="mean").fillna(0.0)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)), [str(item) for item in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), pivot.index.tolist())
    ax.set_title("Fig.6 Heterogeneity heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
