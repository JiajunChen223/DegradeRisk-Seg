from __future__ import annotations

import json

import numpy as np
import pandas as pd
import _bootstrap  # noqa: F401

from scripts.common import (
    build_model_and_device,
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
from src.risk_control.selective import apply_global_threshold, risk_coverage_curve, select_global_thresholds, summarize_retention
from src.utils.experiment import save_metrics_summary


def _analysis_config(config: dict) -> dict:
    return dict(config.get("analysis", {}).get("risk_score_comparison", {}))


def _scores(score_name: str, logits, losses: np.ndarray) -> tuple[np.ndarray, bool]:
    if score_name == "oracle":
        return -np.asarray(losses, dtype=np.float32), True
    return patch_confidence_scores(logits, score_name), False


def main() -> None:
    args = parse_args("Compare selective risk scores against an oracle upper bound")
    config, run_dir = load_experiment(args)
    maybe_prepare_data(config)
    model, device = build_model_and_device(config, args.device)
    checkpoint_source = load_checkpoint(model, config)
    analysis_cfg = _analysis_config(config)
    score_names = [str(item) for item in analysis_cfg.get("scores", ["msp", "margin", "entropy", "oracle"])]
    target_coverages = [float(item) for item in analysis_cfg.get("target_coverages", config["risk_control"]["target_coverages"])]
    use_calibrated_logits = bool(analysis_cfg.get("use_calibrated_logits", True))

    family = str(analysis_cfg.get("eval_family", config["degradation"].get("default_eval_family", "clean")))
    severity = str(analysis_cfg.get("eval_severity", config["degradation"].get("severity", "L2")))
    degrader = None if family == "clean" else build_degrader(config, family_name=family, severity_name=severity)
    calibrator = None
    calibrator_source = None
    if use_calibrated_logits:
        calibrator, calibrator_source = load_calibrator_from_config(config)

    calib_loader = make_loader(config, config["split"]["val_calib"], degrader=degrader, shuffle=False)
    test_loader = make_loader(config, config["split"]["test"], degrader=degrader, shuffle=False)
    calib_outputs = run_model_inference(model, calib_loader, device, config=config, calibrator=calibrator)
    test_outputs = run_model_inference(model, test_loader, device, config=config, calibrator=calibrator)

    summary_rows = []
    curve_rows = []
    for score_name in score_names:
        calib_scores, calib_uses_gt = _scores(score_name, calib_outputs["logits"], calib_outputs["dice_losses"])
        test_scores, test_uses_gt = _scores(score_name, test_outputs["logits"], test_outputs["dice_losses"])
        threshold_map = select_global_thresholds(calib_scores, target_coverages)
        curve = risk_coverage_curve(test_scores, test_outputs["dice_losses"])
        for idx, (coverage, risk) in enumerate(zip(curve["coverage"].tolist(), curve["risk"].tolist()), start=1):
            curve_rows.append(
                {
                    "protocol_id": config["experiment"]["protocol_id"],
                    "run_name": config["experiment"]["run_name"],
                    "feature_set": config["data"]["name"],
                    "arm_role": config["experiment"].get("arm_role"),
                    "eval_family": family,
                    "eval_severity": severity,
                    "score_name": score_name,
                    "uses_gt_for_calibration_score": bool(calib_uses_gt),
                    "uses_gt_for_test_score": bool(test_uses_gt),
                    "curve_point_index": idx,
                    "coverage": float(coverage),
                    "retained_risk": float(risk),
                }
            )
        row = {
            "protocol_id": config["experiment"]["protocol_id"],
            "run_name": config["experiment"]["run_name"],
            "feature_set": config["data"]["name"],
            "arm_role": config["experiment"].get("arm_role"),
            "eval_family": family,
            "eval_severity": severity,
            "score_name": score_name,
            "uses_calibrated_logits": use_calibrated_logits,
            "uses_gt_for_calibration_score": bool(calib_uses_gt),
            "uses_gt_for_test_score": bool(test_uses_gt),
            "aurc": float(curve["aurc"]),
        }
        for target_coverage in target_coverages:
            coverage_key = f"{target_coverage:.2f}"
            threshold = float(threshold_map[coverage_key])
            keep = apply_global_threshold(test_scores, threshold)
            stats = summarize_retention(test_outputs["dice_losses"], keep)
            row[f"threshold_{coverage_key}"] = threshold
            row[f"coverage_{coverage_key}"] = stats["coverage"]
            row[f"risk_{coverage_key}"] = stats["retained_risk"]
        summary_rows.append(row)

    pd.DataFrame(summary_rows).to_csv(run_dir / "tables" / "risk_score_comparison.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(run_dir / "tables" / "risk_score_curves.csv", index=False)
    summary_frame = pd.DataFrame(summary_rows)
    non_oracle = summary_frame.loc[~summary_frame["uses_gt_for_test_score"]].copy()
    payload = {
        **summary_provenance(config, checkpoint_source=checkpoint_source, calibrator_source=calibrator_source),
        "analysis_kind": "risk_score_comparison",
        "eval_family": family,
        "eval_severity": severity,
        "score_names": score_names,
        "target_coverages": target_coverages,
        "best_non_oracle_score_by_aurc": str(non_oracle.sort_values("aurc").iloc[0]["score_name"]) if not non_oracle.empty else None,
        "oracle_aurc": float(summary_frame.loc[summary_frame["score_name"] == "oracle", "aurc"].iloc[0]) if "oracle" in set(summary_frame["score_name"]) else None,
    }
    save_metrics_summary(payload, run_dir)
    (run_dir / "risk_score_comparison.log").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
