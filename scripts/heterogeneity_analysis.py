from __future__ import annotations

import json

import numpy as np
import pandas as pd
import _bootstrap  # noqa: F401

from scripts.common import (
    build_model_and_device,
    load_checkpoint,
    load_experiment,
    make_loader,
    maybe_prepare_data,
    parse_args,
    run_model_inference,
    summary_provenance,
)
from src.analysis.heterogeneity import (
    assign_temporal_cluster,
    build_group_record,
    fit_analysis_bin_edges,
    fit_temporal_clusterer,
    summarize_group_results,
    worst_group_summary,
)
from src.degradations.registry import build_degrader
from src.evaluators.calibration import compute_binary_calibration_metrics
from src.evaluators.segmentation import batch_dice_iou
from src.utils.experiment import save_metrics_summary
from src.visualization.plot_main_figures import plot_heterogeneity_heatmap


def main() -> None:
    args = parse_args("Endogenous heterogeneity analysis")
    config, run_dir = load_experiment(args)
    maybe_prepare_data(config)
    model, device = build_model_and_device(config, args.device)
    checkpoint_source = load_checkpoint(model, config)

    train_loader = make_loader(config, config["split"]["train"])
    train_outputs = run_model_inference(model, train_loader, device)
    curves = [item["input_safe"]["temporal_curve"] for item in train_outputs["descriptors"]]
    clusterer = fit_temporal_clusterer(curves, n_clusters=int(config["analysis"]["temporal_clustering"]["n_clusters"]), seed=int(config.get("seed", 1)))
    analysis_edges = fit_analysis_bin_edges(train_outputs["descriptors"])

    family = config["degradation"].get("default_eval_family", "clean")
    severity = config["degradation"].get("severity", "L2")
    degrader = None if family == "clean" else build_degrader(config, family_name=family, severity_name=severity)
    test_loader = make_loader(config, config["split"]["test"], degrader=degrader)
    outputs = run_model_inference(model, test_loader, device)
    records = []
    for sample_id, descriptor, meta, logit, target in zip(
        outputs["sample_ids"],
        outputs["descriptors"],
        outputs["metas"],
        outputs["logits"],
        outputs["targets"],
    ):
        cluster_id = assign_temporal_cluster(clusterer, descriptor["input_safe"]["temporal_curve"])
        seg = batch_dice_iou(logit.unsqueeze(0), target)
        cal = compute_binary_calibration_metrics(logit.unsqueeze(0), target.unsqueeze(0))
        metrics = {
            "dice": seg["dice"],
            "iou": seg["iou"],
            "ece": cal["ece"],
            "retained_risk": 1.0 - seg["dice"],
        }
        records.append(
            build_group_record(
                sample_id,
                descriptor,
                metrics,
                meta,
                temporal_cluster=cluster_id,
                analysis_edges=analysis_edges,
            )
        )
    significance_cfg = config["analysis"]["significance"]
    summary = summarize_group_results(
        records,
        rounds=int(significance_cfg.get("bootstrap_rounds", 1000)),
        confidence=float(significance_cfg.get("confidence", 0.95)),
        seed=int(config.get("seed", 1)),
    )
    summary.to_csv(run_dir / "tables" / "heterogeneity_summary.csv", index=False)
    heatmap_frame = summary.loc[summary["stratum"] == "temporal_cluster"].rename(columns={"group_label": "temporal_cluster"})
    if not heatmap_frame.empty:
        plot_heterogeneity_heatmap(heatmap_frame, run_dir / "figures" / "fig6_heterogeneity.png", value="retained_risk")
    merged = worst_group_summary(summary)
    merged.to_csv(run_dir / "tables" / "worst_group_summary.csv", index=False)
    risk_row = merged.sort_values(["worst_group_risk", "worst_group_risk_count"], ascending=[False, False]).iloc[0]
    ece_row = merged.sort_values(["worst_group_ece", "worst_group_ece_count"], ascending=[False, False]).iloc[0]
    dice_row = merged.sort_values(["worst_group_dice", "worst_group_dice_count"], ascending=[True, False]).iloc[0]
    payload = {
        **summary_provenance(config, checkpoint_source=checkpoint_source),
        "family": family,
        "severity": severity,
        "worst_group_risk": float(risk_row["worst_group_risk"]),
        "worst_group_risk_count": int(risk_row["worst_group_risk_count"]),
        "worst_group_ece": float(ece_row["worst_group_ece"]),
        "worst_group_ece_count": int(ece_row["worst_group_ece_count"]),
        "worst_group_dice": float(dice_row["worst_group_dice"]),
        "worst_group_dice_count": int(dice_row["worst_group_dice_count"]),
        "bootstrap_rounds": int(significance_cfg.get("bootstrap_rounds", 1000)),
        "risk_ci_low": float(risk_row["worst_group_risk_ci_low"]),
        "risk_ci_high": float(risk_row["worst_group_risk_ci_high"]),
    }
    save_metrics_summary(payload, run_dir)
    (run_dir / "heterogeneity.log").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
