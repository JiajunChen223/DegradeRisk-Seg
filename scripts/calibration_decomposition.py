from __future__ import annotations

import json

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
    resolve_degradation_grid,
    run_model_inference,
    summary_provenance,
)
from src.calibration.temperature_scaling import apply_calibration
from src.degradations.registry import build_degrader
from src.evaluators.calibration import calibration_bin_table, compute_binary_calibration_metrics, compute_calibration_decomposition
from src.utils.experiment import save_metrics_summary


def _analysis_config(config: dict) -> dict:
    return dict(config.get("analysis", {}).get("calibration_decomposition", {}))


def _calibrated_logits(outputs: dict, calibrator, groupwise: bool, group_keys: list[str] | None):
    if calibrator is None:
        return None
    groups = descriptor_groups(outputs["descriptors"], group_keys=group_keys) if groupwise else None
    return apply_calibration(outputs["logits"], calibrator, groups)


def main() -> None:
    args = parse_args("Calibration decomposition under degraded observations")
    config, run_dir = load_experiment(args)
    maybe_prepare_data(config)
    model, device = build_model_and_device(config, args.device)
    checkpoint_source = load_checkpoint(model, config)
    analysis_cfg = _analysis_config(config)
    bins = int(analysis_cfg.get("bins", 15))
    high_confidence_threshold = float(analysis_cfg.get("high_confidence_threshold", 0.90))
    include_raw = bool(analysis_cfg.get("include_raw", True))
    include_calibrated = bool(analysis_cfg.get("include_calibrated", True))

    calibrator = None
    calibrator_source = None
    if include_calibrated:
        calibrator, calibrator_source = load_calibrator_from_config(config)
    calibration_mode = str(config.get("calibration", {}).get("mode", "global"))
    groupwise = calibration_mode == "descriptor_group"
    group_keys = config.get("calibration", {}).get("group_keys") if groupwise else None

    summary_rows = []
    bin_rows = []
    families, severity_map = resolve_degradation_grid(config, analysis_cfg)
    for family in families:
        for severity in severity_map[family]:
            degrader = None if family == "clean" else build_degrader(config, family_name=family, severity_name=severity)
            loader = make_loader(config, str(analysis_cfg.get("split", config["split"]["test"])), degrader=degrader, shuffle=False)
            outputs = run_model_inference(model, loader, device, config=config, group_keys=group_keys)
            cases = []
            if include_raw:
                cases.append(("raw", outputs["logits"]))
            calibrated = _calibrated_logits(outputs, calibrator, groupwise=groupwise, group_keys=group_keys)
            if calibrated is not None:
                cases.append(("calibrated", calibrated))
            for case_name, logits in cases:
                metrics = compute_binary_calibration_metrics(logits, outputs["targets"], bins=bins)
                decomposition = compute_calibration_decomposition(
                    logits,
                    outputs["targets"],
                    bins=bins,
                    high_confidence_threshold=high_confidence_threshold,
                )
                summary_rows.append(
                    {
                        "protocol_id": config["experiment"]["protocol_id"],
                        "run_name": config["experiment"]["run_name"],
                        "feature_set": config["data"]["name"],
                        "arm_role": config["experiment"].get("arm_role"),
                        "eval_family": family,
                        "eval_severity": severity,
                        "calibration_case": case_name,
                        "calibration_mode": calibration_mode if case_name == "calibrated" else "none",
                        "bins": bins,
                        **metrics,
                        **decomposition,
                    }
                )
                for row in calibration_bin_table(logits, outputs["targets"], bins=bins):
                    bin_rows.append(
                        {
                            "protocol_id": config["experiment"]["protocol_id"],
                            "run_name": config["experiment"]["run_name"],
                            "feature_set": config["data"]["name"],
                            "arm_role": config["experiment"].get("arm_role"),
                            "eval_family": family,
                            "eval_severity": severity,
                            "calibration_case": case_name,
                            **row,
                        }
                    )

    pd.DataFrame(summary_rows).to_csv(run_dir / "tables" / "calibration_decomposition.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(run_dir / "tables" / "calibration_bins.csv", index=False)
    summary_frame = pd.DataFrame(summary_rows)
    payload = {
        **summary_provenance(config, checkpoint_source=checkpoint_source, calibrator_source=calibrator_source),
        "analysis_kind": "calibration_decomposition",
        "calibration_decomposition_rows": int(len(summary_rows)),
        "calibration_bin_rows": int(len(bin_rows)),
        "calibration_cases": sorted(summary_frame["calibration_case"].unique().tolist()) if not summary_frame.empty else [],
        "mean_ece": float(summary_frame["ece"].mean()) if not summary_frame.empty else 0.0,
        "mean_overconfidence_ece": float(summary_frame["overconfidence_ece"].mean()) if not summary_frame.empty else 0.0,
        "mean_underconfidence_ece": float(summary_frame["underconfidence_ece"].mean()) if not summary_frame.empty else 0.0,
    }
    save_metrics_summary(payload, run_dir)
    (run_dir / "calibration_decomposition.log").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
