from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SUMMARY_SCHEMA_VERSION = "paper_v2"

FINAL_PROVENANCE_FIELDS = [
    "summary_schema_version",
    "run_name",
    "protocol_id",
    "arm_role",
    "feature_set",
    "feature_family_count",
    "tensor_channel_count",
    "seed",
    "config_path",
]


def _has_non_empty_value(row: pd.Series, field: str) -> bool:
    if field not in row.index:
        return False
    value = row[field]
    if pd.isna(value):
        return False
    text = str(value).strip()
    return len(text) > 0


def _is_namespaced_artifact(value: Any) -> bool:
    text = str(value)
    return "outputs" in text and "checkpoints" in text and "seed" in text


def _table_has_columns(
    run_dir: str | Path,
    relative_path: str,
    required_cols: list[str],
    forbidden_cols: list[str] | None = None,
) -> bool:
    table_path = Path(run_dir) / relative_path
    if not table_path.exists():
        return False
    try:
        header = pd.read_csv(table_path, nrows=0).columns.tolist()
    except Exception:
        return False
    if not all(col in header for col in required_cols):
        return False
    if forbidden_cols and any(col in header for col in forbidden_cols):
        return False
    return True


def _read_table_with_run_metadata(row: pd.Series, relative_path: str) -> pd.DataFrame | None:
    table_path = Path(str(row["run_dir"])) / relative_path
    if not table_path.exists():
        return None
    table = pd.read_csv(table_path)
    table["protocol_id"] = row.get("protocol_id")
    table["run_name"] = row.get("run_name")
    table["feature_set"] = row.get("feature_set")
    table["arm_role"] = row.get("arm_role")
    table["seed"] = row.get("seed")
    if _has_non_empty_value(row, "eval_family"):
        table["eval_family"] = row.get("eval_family")
    if _has_non_empty_value(row, "eval_severity"):
        table["eval_severity"] = row.get("eval_severity")
    return table


def collect_run_summaries(runs_root: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metrics_file in Path(runs_root).glob("*/*/metrics_summary.json"):
        payload = json.loads(metrics_file.read_text(encoding="utf-8"))
        payload["run_dir"] = str(metrics_file.parent)
        rows.append(payload)
    return pd.DataFrame(rows)


def filter_final_paper_runs(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if any(field not in frame.columns for field in FINAL_PROVENANCE_FIELDS):
        return frame.iloc[0:0].copy()

    keep_mask = []
    for _, row in frame.iterrows():
        valid = all(_has_non_empty_value(row, field) for field in FINAL_PROVENANCE_FIELDS)
        if valid:
            valid = str(row.get("summary_schema_version", "")) == SUMMARY_SCHEMA_VERSION
        protocol_id = str(row.get("protocol_id", ""))
        run_dir = row.get("run_dir", "")
        if valid and protocol_id == "P00":
            valid = all(_has_non_empty_value(row, field) for field in ["max_epochs", "min_epochs", "effective_batch_size", "checkpoint_source"])
            if valid:
                valid = _is_namespaced_artifact(row["checkpoint_source"])
        if valid and protocol_id in {"P02", "P04", "P05", "P06", "P07", "P10"}:
            valid = _has_non_empty_value(row, "checkpoint_source") and _is_namespaced_artifact(row["checkpoint_source"])
        if valid and protocol_id in {"P02", "P10"}:
            valid = _table_has_columns(
                run_dir,
                "tables/degradation_summary.csv",
                required_cols=["ordinal_severity_family", "severity_semantics", "dice", "ece", "feature_set", "arm_role"],
            )
        if valid and protocol_id == "P04":
            valid = all(_has_non_empty_value(row, field) for field in ["calibration_fit_split", "calibration_report_split"])
        if valid and protocol_id in {"P05", "P06"}:
            valid = _has_non_empty_value(row, "calibrator_source") and _is_namespaced_artifact(row["calibrator_source"])
            if valid:
                valid = _table_has_columns(
                    run_dir,
                    "tables/risk_coverage_summary.csv",
                    required_cols=["protocol_id", "run_name", "feature_set", "arm_role", "eval_family", "eval_severity", "coverage", "retained_risk"],
                )
            if valid:
                valid = _table_has_columns(
                    run_dir,
                    "tables/risk_coverage_curve.csv",
                    required_cols=["protocol_id", "run_name", "feature_set", "arm_role", "eval_family", "eval_severity", "coverage", "retained_risk"],
                )
        if valid and protocol_id == "P06":
            valid = _has_non_empty_value(row, "descriptor_schema_version") and ("descriptor_group_keys" in row.index)
            if valid:
                valid = _table_has_columns(
                    run_dir,
                    "tables/group_risk_summary.csv",
                    required_cols=["protocol_id", "run_name", "feature_set", "arm_role", "eval_family", "eval_severity", "target_coverage", "group_label", "group_count", "group_retained_risk"],
                )
        if valid and protocol_id == "P07":
            valid = _table_has_columns(
                run_dir,
                "tables/worst_group_summary.csv",
                required_cols=["family", "severity", "worst_group_risk", "worst_group_ece", "worst_group_dice", "worst_group_risk_count"],
            )
        keep_mask.append(bool(valid))
    return frame.loc[keep_mask].reset_index(drop=True)


def require_final_paper_runs(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = filter_final_paper_runs(frame)
    if filtered.empty:
        raise ValueError(
            "No final-paper runs matched the current artifact schema. "
            "Only summary_schema_version=paper_v2 runs with complete provenance are eligible."
        )
    return filtered


def validate_paper_dependencies(frame: pd.DataFrame) -> pd.DataFrame:
    filtered = require_final_paper_runs(frame)
    p05 = filtered.loc[filtered["protocol_id"] == "P05"].copy()
    p06 = filtered.loc[filtered["protocol_id"] == "P06"].copy()
    if p06.empty:
        return filtered
    match_cols = [col for col in ["feature_set", "arm_role", "seed", "eval_family", "eval_severity"] if col in filtered.columns]
    available = {tuple(str(row[col]) for col in match_cols) for _, row in p05.iterrows()}
    missing = []
    for _, row in p06.iterrows():
        identity = tuple(str(row[col]) for col in match_cols)
        if identity not in available:
            missing.append(str(row.get("run_name", row.get("experiment_id", "P06"))))
    if missing:
        raise ValueError(
            "P06 descriptor-conditioned runs require matching P05 global baselines. Missing P05 for: "
            + ", ".join(missing)
        )
    return filtered


def build_table3_degradation_benchmark(frame: pd.DataFrame) -> pd.DataFrame:
    tables = []
    subset = validate_paper_dependencies(frame)
    subset = subset.loc[subset["protocol_id"].isin(["P02", "P10"])].copy()
    for _, row in subset.iterrows():
        table = _read_table_with_run_metadata(row, "tables/degradation_summary.csv")
        if table is not None:
            tables.append(table)
    if tables:
        merged = pd.concat(tables, ignore_index=True)
        cols = [
            col
            for col in [
                "family",
                "severity",
                "ordinal_severity_family",
                "severity_semantics",
                "degradation_eval_scope",
                "degradation_eval_families",
                "degradation_eval_severities",
                "dice",
                "iou",
                "ece",
                "nll",
                "brier",
            ]
            if col in merged.columns
        ]
        metadata_cols = [col for col in ["protocol_id", "run_name", "feature_set", "arm_role", "seed"] if col in merged.columns]
        return merged[metadata_cols + cols].sort_values(metadata_cols + ["family", "severity"]).reset_index(drop=True)
    cols = [col for col in ["family", "severity", "ordinal_severity_family", "dice", "iou"] if col in subset.columns]
    return subset[cols].sort_values(["family", "severity"]) if cols else subset


def build_table4_risk_control(frame: pd.DataFrame) -> pd.DataFrame:
    subset = validate_paper_dependencies(frame)
    subset = subset.loc[subset["protocol_id"].isin(["P04", "P05", "P06"])].copy()
    keep = [
        col
        for col in [
            "protocol_id",
            "run_name",
            "feature_set",
            "arm_role",
            "seed",
            "eval_family",
            "eval_severity",
            "calibration_fit_split",
            "calibration_report_split",
            "calibration_mode",
            "risk_mode",
            "required_baseline_protocol",
            "descriptor_schema_version",
            "descriptor_group_keys",
            "checkpoint_source",
            "calibrator_source",
            "ece",
            "nll",
            "brier",
            "aurc",
            "coverage_0.90_risk",
            "coverage_0.80_risk",
            "group_count",
            "worst_group_target_coverage",
            "worst_group_label",
            "worst_group_retained_risk",
            "worst_group_count",
            "worst_group_risk_gap",
            "paper_operating_points",
            "fig5_semantics",
            "table4_semantics",
            "fig5_curve_source_table",
        ]
        if col in subset.columns
    ]
    return subset[keep] if keep else subset


def build_table5_heterogeneity(frame: pd.DataFrame) -> pd.DataFrame:
    tables = []
    subset = validate_paper_dependencies(frame)
    subset = subset.loc[subset["protocol_id"] == "P07"].copy()
    for _, row in subset.iterrows():
        table = _read_table_with_run_metadata(row, "tables/worst_group_summary.csv")
        if table is not None:
            tables.append(table)
    if tables:
        merged = pd.concat(tables, ignore_index=True)
        cols = [
            col
            for col in [
                "protocol_id",
                "run_name",
                "feature_set",
                "arm_role",
                "seed",
                "family",
                "severity",
                "worst_group_risk",
                "worst_group_risk_stratum",
                "worst_group_risk_label",
                "worst_group_risk_count",
                "worst_group_risk_ci_low",
                "worst_group_risk_ci_high",
                "worst_group_ece",
                "worst_group_ece_stratum",
                "worst_group_ece_label",
                "worst_group_ece_count",
                "worst_group_ece_ci_low",
                "worst_group_ece_ci_high",
                "worst_group_dice",
                "worst_group_dice_stratum",
                "worst_group_dice_label",
                "worst_group_dice_count",
                "worst_group_dice_ci_low",
                "worst_group_dice_ci_high",
            ]
            if col in merged.columns
        ]
        return merged[cols]
    cols = [col for col in ["family", "severity", "worst_group_risk", "worst_group_ece", "worst_group_dice"] if col in subset.columns]
    return subset[cols] if cols else subset
