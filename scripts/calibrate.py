from __future__ import annotations

import json

import numpy as np
import _bootstrap  # noqa: F401

from scripts.common import (
    build_model_and_device,
    calibrator_artifact_path,
    descriptor_groups,
    load_checkpoint,
    load_experiment,
    make_loader,
    maybe_prepare_data,
    parse_args,
    run_model_inference,
    save_calibrator_state,
    summary_provenance,
)
from src.calibration.temperature_scaling import GroupedTemperatureScaler, TemperatureScaler
from src.degradations.registry import build_degrader
from src.evaluators.calibration import compute_binary_calibration_metrics
from src.utils.experiment import save_metrics_summary
from src.visualization.plot_main_figures import plot_reliability_diagram


def main() -> None:
    args = parse_args("Fit post-hoc calibration")
    config, run_dir = load_experiment(args)
    maybe_prepare_data(config)
    model, device = build_model_and_device(config, args.device)
    checkpoint_source = load_checkpoint(model, config)
    family = config["degradation"].get("default_eval_family", "clean")
    severity = config["degradation"].get("severity", "L2")
    mode = config["calibration"]["mode"]
    fit_split = str(config["calibration"].get("fit_split", config["split"].get("val_calib", "val_calib")))
    report_split = str(config["calibration"].get("report_split", config["split"].get("test", "test")))
    degrader = None if family == "clean" else build_degrader(config, family_name=family, severity_name=severity)
    fit_loader = make_loader(config, fit_split, degrader=degrader, shuffle=False)
    report_loader = make_loader(config, report_split, degrader=degrader, shuffle=False)
    group_keys = config["calibration"].get("group_keys") if mode == "descriptor_group" else None
    fit_outputs = run_model_inference(model, fit_loader, device, group_keys=group_keys)
    fit_logits = fit_outputs["logits"]
    fit_targets = fit_outputs["targets"]
    if mode == "global":
        calibrator = TemperatureScaler().fit(fit_logits, fit_targets)
        payload = {"type": "global", **calibrator.state_dict()}
    elif mode == "descriptor_group":
        fit_groups = descriptor_groups(fit_outputs["descriptors"], group_keys=group_keys)
        calibrator = GroupedTemperatureScaler().fit(fit_logits, fit_targets, fit_groups)
        payload = {"type": "descriptor_group", **calibrator.state_dict()}
    else:
        raise ValueError(f"Unsupported calibration mode: {mode}")
    calibrator_source = calibrator_artifact_path(config, family=family, severity=severity)
    save_calibrator_state(payload, run_dir / "calibrator.json")
    save_calibrator_state(payload, calibrator_source)
    report_outputs = run_model_inference(
        model,
        report_loader,
        device,
        calibrator=calibrator,
        groupwise=mode == "descriptor_group",
        group_keys=group_keys,
    )
    report_logits = report_outputs["logits"]
    report_targets = report_outputs["targets"]
    metrics = compute_binary_calibration_metrics(report_logits, report_targets)
    probs = 1.0 / (1.0 + np.exp(-report_logits.numpy().reshape(-1)))
    confidence = np.maximum(probs, 1.0 - probs)
    correctness = ((probs >= 0.5).astype(np.float32) == report_targets.numpy().reshape(-1)).astype(np.float32)
    plot_reliability_diagram(confidence, correctness, run_dir / "figures" / "fig4_reliability.png")
    summary = {
        **summary_provenance(config, checkpoint_source=checkpoint_source, calibrator_source=calibrator_source),
        "calibration_mode": mode,
        "descriptor_group_keys": list(group_keys or []),
        "calibration_fit_split": fit_split,
        "calibration_report_split": report_split,
        "eval_family": family,
        "eval_severity": severity,
        "ece": metrics["ece"],
        "nll": metrics["nll"],
        "brier": metrics["brier"],
    }
    save_metrics_summary(summary, run_dir)
    (run_dir / "eval.log").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
