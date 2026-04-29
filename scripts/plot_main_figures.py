from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import _bootstrap  # noqa: F401

from scripts.common import load_experiment, parse_args
from src.analysis.summary_tables import collect_run_summaries, validate_paper_dependencies
from src.utils.experiment import project_root
from src.visualization.plot_main_figures import (
    plot_modality_drop_summary,
    plot_protocol_matrix,
    plot_risk_coverage_comparison,
    plot_severity_curves,
)


def _normalize_descriptor_group_keys(values: pd.Series) -> list[str]:
    keys: list[str] = []
    for value in values.tolist():
        if isinstance(value, list):
            keys.extend(str(item) for item in value)
        elif isinstance(value, str) and value.strip():
            text = value.strip()
            if text.startswith("[") and text.endswith("]"):
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    parsed = []
                if isinstance(parsed, list):
                    keys.extend(str(item) for item in parsed)
    return sorted(set(keys))


def _aggregate_full_curves(frame: pd.DataFrame, grid_size: int = 101) -> tuple[pd.Series, pd.Series]:
    if frame.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    grid = pd.Series(np.linspace(0.0, 1.0, grid_size), name="coverage")
    run_cols = [col for col in ["run_name", "seed"] if col in frame.columns]
    if not run_cols:
        run_cols = ["protocol_id"]
    curves = []
    for _, run_frame in frame.groupby(run_cols):
        ordered = run_frame.sort_values("coverage")
        coverage = ordered["coverage"].to_numpy(dtype=float)
        risk = ordered["retained_risk"].to_numpy(dtype=float)
        if len(coverage) == 0:
            continue
        interpolated = np.interp(grid.to_numpy(dtype=float), coverage, risk, left=risk[0], right=risk[-1])
        curves.append(interpolated)
    if not curves:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    mean_risk = pd.Series(np.vstack(curves).mean(axis=0), name="retained_risk")
    return grid, mean_risk


def main() -> None:
    args = parse_args("Generate paper figures")
    config, run_dir = load_experiment(args)
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    frame = validate_paper_dependencies(collect_run_summaries(project_root() / "outputs" / "runs"))
    figure_metadata: dict[str, object] = {
        "summary_schema_version": "paper_v2",
        "figure_slot_semantics": {
            "P03": "figure slot / figure-generating analysis stage derived from degradation summaries; no standalone run config is required",
        },
        "ordinal_severity_rule": 'family != "modality_drop"',
        "non_ordinal_family": "modality_drop",
        "paper_operating_points": [0.90, 0.80],
    }
    plot_protocol_matrix(run_dir / "figures" / "fig2_protocols.png")
    deg_tables = []
    p03_source_protocols = ["P02", "P10"]
    for _, row in frame.loc[frame["protocol_id"].isin(p03_source_protocols)].iterrows():
        table_path = Path(str(row["run_dir"])) / "tables" / "degradation_summary.csv"
        if not table_path.exists():
            continue
        table = pd.read_csv(table_path)
        table["feature_set"] = row.get("feature_set")
        table["arm_role"] = row.get("arm_role")
        table["protocol_id"] = row.get("protocol_id")
        deg_tables.append(table)
    if deg_tables:
        deg_frame = pd.concat(deg_tables, ignore_index=True)
        plot_severity_curves(
            deg_frame,
            run_dir / "figures" / "fig3_severity.png",
            metric="dice",
            title="Fig.3 Ordinal severity curves (modality_drop shown separately)",
        )
        plot_severity_curves(
            deg_frame,
            run_dir / "figures" / "fig3_severity_ece.png",
            metric="ece",
            title="Fig.3 Ordinal ECE curves (modality_drop shown separately)",
        )
        plot_modality_drop_summary(deg_frame, run_dir / "figures" / "fig3_modality_drop_dice.png", metric="dice")
        figure_metadata["fig3"] = {
            "protocol_id": "P03",
            "semantics": "degradation severity figure slot derived from degradation summaries",
            "source_protocols": p03_source_protocols,
            "severity_curve_families": sorted(str(item) for item in deg_frame.loc[deg_frame["family"] != "modality_drop", "family"].dropna().unique().tolist()),
            "modality_drop_panel": {
                "family": "modality_drop",
                "semantics": "non-ordinal modality-drop panel plotted separately from ordinal severity curves",
            },
            "ordinal_severity_rule": 'family != "modality_drop"',
            "eval_scope": sorted(str(item) for item in deg_frame["degradation_eval_scope"].dropna().unique().tolist()) if "degradation_eval_scope" in deg_frame.columns else [],
        }
    risk_rows = frame.loc[frame["protocol_id"].isin(["P05", "P06"])].copy()
    if not risk_rows.empty:
        curves: dict[str, tuple] = {}
        for slot in ["P05", "P06"]:
            slot_rows = risk_rows.loc[risk_rows["protocol_id"] == slot]
            if slot_rows.empty:
                continue
            tables = []
            for _, row in slot_rows.iterrows():
                table_path = Path(str(row["run_dir"])) / "tables" / "risk_coverage_curve.csv"
                if not table_path.exists():
                    continue
                table = pd.read_csv(table_path)
                table["protocol_id"] = slot
                table["run_name"] = row.get("run_name")
                table["seed"] = row.get("seed")
                tables.append(table)
            if not tables:
                continue
            risk_frame = pd.concat(tables, ignore_index=True)
            coverage, risk = _aggregate_full_curves(risk_frame)
            if coverage.empty or risk.empty:
                continue
            curves["global baseline" if slot == "P05" else "descriptor-conditioned"] = (
                coverage.to_numpy(),
                risk.to_numpy(),
            )
        if "descriptor-conditioned" in curves and "global baseline" not in curves:
            raise ValueError("Fig.5 requires P05 global baseline runs before plotting P06 descriptor-conditioned results.")
        if curves:
            plot_risk_coverage_comparison(
                curves,
                run_dir / "figures" / "fig5_risk_coverage.png",
                title="Fig.5 Full risk-coverage curve",
            )
        figure_metadata["fig5"] = {
            "semantics": "full risk-coverage curve",
            "table4_semantics": "selected operating-point summary",
            "source_protocols": sorted(str(item) for item in risk_rows["protocol_id"].dropna().unique().tolist()),
            "eval_family": sorted(str(item) for item in risk_rows["eval_family"].dropna().unique().tolist()) if "eval_family" in risk_rows.columns else [],
            "eval_severity": sorted(str(item) for item in risk_rows["eval_severity"].dropna().unique().tolist()) if "eval_severity" in risk_rows.columns else [],
            "descriptor_group_keys": _normalize_descriptor_group_keys(risk_rows["descriptor_group_keys"]) if "descriptor_group_keys" in risk_rows.columns else [],
            "paper_operating_points": [0.90, 0.80],
            "curve_source_table": "risk_coverage_curve.csv",
            "curve_aggregation": "interpolated_mean_over_runs",
        }
    (run_dir / "figures" / "figure_metadata.json").write_text(json.dumps(figure_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "plot_summary.log").write_text(json.dumps(figure_metadata, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
