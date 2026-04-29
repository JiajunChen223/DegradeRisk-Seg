from __future__ import annotations

import json

import pandas as pd
import _bootstrap  # noqa: F401

from scripts.common import (
    build_model_and_device,
    load_checkpoint,
    load_experiment,
    make_loader,
    maybe_prepare_data,
    parse_args,
    prepare_model_inputs,
    summary_provenance,
)
from src.degradations.registry import build_degrader
from src.evaluators.calibration import compute_binary_calibration_metrics
from src.evaluators.robustness import summarize_degradation_results
from src.evaluators.segmentation import batch_dice_iou
from src.utils.experiment import save_metrics_summary


def main() -> None:
    args = parse_args("Evaluate Plot-Rice robustness")
    config, run_dir = load_experiment(args)
    maybe_prepare_data(config)
    model, device = build_model_and_device(config, args.device)
    checkpoint_source = load_checkpoint(model, config)

    rows = []
    degradation_cfg = config["degradation"]
    if degradation_cfg.get("enabled", False):
        families = degradation_cfg["eval_families"]
        severity_map = {family: list(degradation_cfg["families"][family]["severities"].keys()) for family in families}
    else:
        families = ["clean"]
        severity_map = {"clean": ["L0"]}

    for family in families:
        for severity in severity_map[family]:
            degrader = None if family == "clean" else build_degrader(config, family_name=family, severity_name=severity)
            loader = make_loader(config, config["split"]["test"], degrader=degrader, shuffle=False)
            infer_rows = []
            all_logits = []
            all_targets = []
            for batch in loader:
                import torch

                x = prepare_model_inputs(batch["x"], device, config)
                y = batch["y"].to(device, non_blocking=True)
                with torch.no_grad():
                    logits = model(x)
                seg = batch_dice_iou(logits, y)
                cal = compute_binary_calibration_metrics(logits, y.unsqueeze(1))
                for sample_id in batch["sample_id"]:
                    infer_rows.append({"sample_id": sample_id, "family": family, "severity": severity, **seg, **cal})
                all_logits.append(logits.detach().cpu())
                all_targets.append(y.detach().cpu().unsqueeze(1))
            rows.extend(infer_rows)

    summary_frame = summarize_degradation_results(rows)
    if not summary_frame.empty:
        eval_scope = "family_severity_grid"
        eval_families = list(families)
        eval_severities = severity_map
        summary_frame["ordinal_severity_family"] = summary_frame["family"] != "modality_drop"
        summary_frame["severity_semantics"] = summary_frame["ordinal_severity_family"].map(
            lambda is_ordinal: "ordinal" if bool(is_ordinal) else "non_ordinal_modality_drop"
        )
        summary_frame["degradation_eval_scope"] = eval_scope
        summary_frame["degradation_eval_families"] = json.dumps(eval_families, ensure_ascii=False)
        summary_frame["degradation_eval_severities"] = json.dumps(eval_severities, ensure_ascii=False)
        summary_frame["protocol_id"] = config["experiment"]["protocol_id"]
        summary_frame["run_name"] = config["experiment"]["run_name"]
        summary_frame["feature_set"] = config["data"]["name"]
        summary_frame["arm_role"] = config["experiment"].get("arm_role")
    summary_frame.to_csv(run_dir / "tables" / "degradation_summary.csv", index=False)
    avg_degraded_dice = float(summary_frame["dice"].mean()) if not summary_frame.empty else 0.0
    payload = {
        **summary_provenance(config, checkpoint_source=checkpoint_source),
        "avg_degraded_dice": avg_degraded_dice,
        "family": "aggregate",
        "severity": "all",
        "dice": avg_degraded_dice,
        "iou": float(summary_frame["iou"].mean()) if not summary_frame.empty else 0.0,
        "eval_families": list(families),
        "severity_levels": severity_map,
        "degradation_eval_scope": "family_severity_grid",
        "degradation_eval_families": list(families),
        "degradation_eval_severities": severity_map,
        "ordinal_severity_rule": 'family != "modality_drop"',
    }
    save_metrics_summary(payload, run_dir)
    (run_dir / "eval.log").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
