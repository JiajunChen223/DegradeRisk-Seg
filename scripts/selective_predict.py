from __future__ import annotations

import json

import numpy as np
import pandas as pd
import _bootstrap  # noqa: F401

from scripts.common import (
    build_model_and_device,
    descriptor_groups,
    load_calibrator_from_config,
    load_checkpoint,
    load_experiment,
    make_loader,
    maybe_prepare_data,
    parse_args,
    run_model_inference,
    summary_provenance,
)
from src.degradations.registry import build_degrader
from src.risk_control.confidence_scores import patch_confidence_scores
from src.risk_control.selective import (
    apply_global_threshold,
    apply_groupwise_threshold,
    risk_coverage_curve,
    select_global_thresholds,
    select_groupwise_thresholds,
    summarize_group_retention,
    summarize_retention,
    worst_group_retention,
)
from src.utils.experiment import save_metrics_summary
from src.visualization.plot_main_figures import plot_risk_coverage


def main() -> None:
    args = parse_args("Selective prediction under degraded observations")
    config, run_dir = load_experiment(args)
    maybe_prepare_data(config)
    model, device = build_model_and_device(config, args.device)
    checkpoint_source = load_checkpoint(model, config)
    family = config["degradation"].get("default_eval_family", "clean")
    severity = config["degradation"].get("severity", "L2")
    degrader = None if family == "clean" else build_degrader(config, family_name=family, severity_name=severity)
    try:
        calibrator, calibrator_source = load_calibrator_from_config(config)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "No calibrator could be resolved for selective prediction. "
            "Run calibrate.py for the configured calibration entry first."
        ) from exc
    groupwise = config["risk_control"]["threshold_mode"] == "descriptor_group"
    group_keys = config["risk_control"].get("group_keys") if groupwise else None

    calib_loader = make_loader(config, config["split"]["val_calib"], degrader=degrader, shuffle=False)
    calib_outputs = run_model_inference(
        model,
        calib_loader,
        device,
        calibrator=calibrator,
        groupwise=groupwise,
        group_keys=group_keys,
    )
    calib_scores = patch_confidence_scores(calib_outputs["logits"], config["risk_control"]["score"])
    calib_losses = calib_outputs["dice_losses"]
    calib_groups = descriptor_groups(calib_outputs["descriptors"], group_keys=group_keys)

    threshold_map = (
        select_groupwise_thresholds(calib_scores, calib_groups, config["risk_control"]["target_coverages"])
        if groupwise
        else select_global_thresholds(calib_scores, config["risk_control"]["target_coverages"])
    )

    test_loader = make_loader(config, config["split"]["test"], degrader=degrader, shuffle=False)
    test_outputs = run_model_inference(
        model,
        test_loader,
        device,
        calibrator=calibrator,
        groupwise=groupwise,
        group_keys=group_keys,
    )
    test_scores = patch_confidence_scores(test_outputs["logits"], config["risk_control"]["score"])
    test_losses = test_outputs["dice_losses"]
    test_groups = descriptor_groups(test_outputs["descriptors"], group_keys=group_keys)
    curve = risk_coverage_curve(test_scores, test_losses)
    plot_risk_coverage(curve["coverage"], curve["risk"], run_dir / "figures" / "fig5_risk_coverage.png")
    curve_rows = []
    for idx, (coverage, risk) in enumerate(zip(curve["coverage"].tolist(), curve["risk"].tolist()), start=1):
        curve_rows.append(
            {
                "protocol_id": config["experiment"]["protocol_id"],
                "run_name": config["experiment"]["run_name"],
                "feature_set": config["data"]["name"],
                "arm_role": config["experiment"].get("arm_role"),
                "eval_family": family,
                "eval_severity": severity,
                "threshold_mode": config["risk_control"]["threshold_mode"],
                "curve_point_index": idx,
                "coverage": float(coverage),
                "retained_risk": float(risk),
            }
        )
    rows = []
    group_rows_all = []
    paper_operating_points = [0.90, 0.80]
    primary_worst_group = None
    summary = {
        **summary_provenance(config, checkpoint_source=checkpoint_source, calibrator_source=calibrator_source),
        "risk_mode": config["risk_control"]["threshold_mode"],
        "score_name": config["risk_control"]["score"],
        "threshold_mode": config["risk_control"]["threshold_mode"],
        "calibration_mode": config["calibration"]["mode"],
        "descriptor_group_keys": list(group_keys or []),
        "required_baseline_protocol": "P05" if config["experiment"]["protocol_id"] == "P06" else None,
        "descriptor_schema_version": (
            test_outputs["descriptors"][0]["input_safe"].get("descriptor_schema_version")
            if test_outputs["descriptors"]
            else None
        ),
        "eval_family": family,
        "eval_severity": severity,
        "paper_operating_points": paper_operating_points,
        "fig5_semantics": "full risk-coverage curve",
        "table4_semantics": "selected operating-point summary",
        "fig5_curve_source_table": "tables/risk_coverage_curve.csv",
        "aurc": float(curve["aurc"]),
    }
    if groupwise:
        summary["group_count"] = int(len(set(test_groups)))
    for target_coverage in config["risk_control"]["target_coverages"]:
        coverage_str = f"{float(target_coverage):.2f}"
        if groupwise:
            per_group = {group: float(values[coverage_str]) for group, values in threshold_map.items()}
            keep = apply_groupwise_threshold(test_scores, test_groups, per_group)
            threshold_value = np.nan
        else:
            threshold_value = float(threshold_map[coverage_str])
            keep = apply_global_threshold(test_scores, threshold_value)
        stats = summarize_retention(test_losses, keep)
        rows.append(
            {
                "protocol_id": config["experiment"]["protocol_id"],
                "run_name": config["experiment"]["run_name"],
                "feature_set": config["data"]["name"],
                "arm_role": config["experiment"].get("arm_role"),
                "eval_family": family,
                "eval_severity": severity,
                "threshold_mode": config["risk_control"]["threshold_mode"],
                "target_coverage": float(coverage_str),
                "threshold": threshold_value,
                **stats,
            }
        )
        summary[f"coverage_{coverage_str}_risk"] = stats["retained_risk"]
        if groupwise:
            group_rows = summarize_group_retention(test_losses, keep, test_groups)
            worst_group = worst_group_retention(group_rows)
            summary[f"coverage_{coverage_str}_worst_group_retained_risk"] = worst_group["worst_group_retained_risk"]
            summary[f"coverage_{coverage_str}_worst_group_risk_gap"] = worst_group["worst_group_risk_gap"]
            if coverage_str == "0.90":
                primary_worst_group = worst_group
            for group_row in group_rows:
                group_rows_all.append(
                    {
                        "protocol_id": config["experiment"]["protocol_id"],
                        "run_name": config["experiment"]["run_name"],
                        "feature_set": config["data"]["name"],
                        "arm_role": config["experiment"].get("arm_role"),
                        "eval_family": family,
                        "eval_severity": severity,
                        "threshold_mode": config["risk_control"]["threshold_mode"],
                        "target_coverage": float(coverage_str),
                        "group_label": group_row["group_label"],
                        "group_count": group_row["group_count"],
                        "group_coverage": group_row["group_coverage"],
                        "group_retained_count": group_row["group_retained_count"],
                        "group_retained_risk": group_row["group_retained_risk"],
                    }
                )

    pd.DataFrame(rows).to_csv(run_dir / "tables" / "risk_coverage_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(run_dir / "tables" / "risk_coverage_curve.csv", index=False)
    if group_rows_all:
        pd.DataFrame(group_rows_all).to_csv(run_dir / "tables" / "group_risk_summary.csv", index=False)
    if primary_worst_group is not None:
        summary["worst_group_target_coverage"] = 0.90
        summary["worst_group_label"] = primary_worst_group["worst_group_label"]
        summary["worst_group_retained_risk"] = primary_worst_group["worst_group_retained_risk"]
        summary["worst_group_count"] = primary_worst_group["worst_group_count"]
        summary["worst_group_risk_gap"] = primary_worst_group["worst_group_risk_gap"]
    save_metrics_summary(summary, run_dir)
    (run_dir / "selective_predict.log").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
