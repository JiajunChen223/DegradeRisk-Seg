from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.analysis.heterogeneity import summarize_group_results, worst_group_summary
from src.analysis.summary_tables import (
    SUMMARY_SCHEMA_VERSION,
    build_table3_degradation_benchmark,
    build_table4_risk_control,
    validate_paper_dependencies,
)


class PaperGuardTest(unittest.TestCase):
    def test_p06_requires_matching_p05(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / "P06" / "seed01"
            (run_dir / "tables").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "protocol_id": "P06",
                        "run_name": "P06__core5__arm-designprior__resunet__train-clean__eval-degbench__risk-descgrp",
                        "feature_set": "core5",
                        "arm_role": "design_prior",
                        "eval_family": "random_missing_months",
                        "eval_severity": "L2",
                        "coverage": 0.8,
                        "retained_risk": 0.2,
                    }
                ]
            ).to_csv(run_dir / "tables" / "risk_coverage_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "protocol_id": "P06",
                        "run_name": "P06__core5__arm-designprior__resunet__train-clean__eval-degbench__risk-descgrp",
                        "feature_set": "core5",
                        "arm_role": "design_prior",
                        "eval_family": "random_missing_months",
                        "eval_severity": "L2",
                        "coverage": 0.8,
                        "retained_risk": 0.2,
                    }
                ]
            ).to_csv(run_dir / "tables" / "risk_coverage_curve.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "protocol_id": "P06",
                        "run_name": "P06__core5__arm-designprior__resunet__train-clean__eval-degbench__risk-descgrp",
                        "feature_set": "core5",
                        "arm_role": "design_prior",
                        "eval_family": "random_missing_months",
                        "eval_severity": "L2",
                        "target_coverage": 0.9,
                        "group_label": "mid|short|mid|mid",
                        "group_count": 4,
                        "group_retained_risk": 0.25,
                    }
                ]
            ).to_csv(run_dir / "tables" / "group_risk_summary.csv", index=False)
            frame = pd.DataFrame(
                [
                    {
                        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
                        "protocol_id": "P06",
                        "run_name": "P06__core5__arm-designprior__resunet__train-clean__eval-degbench__risk-descgrp",
                        "arm_role": "design_prior",
                        "feature_set": "core5",
                        "feature_family_count": 5,
                        "tensor_channel_count": 5,
                        "seed": 1,
                        "config_path": "configs/protocols/P06.yaml",
                        "checkpoint_source": "outputs/checkpoints/P00__core5/seed01/checkpoint__best-dice.pt",
                        "calibrator_source": "outputs/checkpoints/P06__core5/seed01/calibrator__mode-descriptor_group__family-random_missing_months__severity-L2.json",
                        "descriptor_schema_version": "input_safe_v3",
                        "descriptor_group_keys": ["budget_bin", "longest_gap_bin"],
                        "eval_family": "random_missing_months",
                        "eval_severity": "L2",
                        "run_dir": str(run_dir),
                    }
                ]
            )
            with self.assertRaises(ValueError):
                validate_paper_dependencies(frame)

    def test_heterogeneity_summary_is_not_sparse_by_default(self) -> None:
        records = [
            {"family": "random_missing_months", "severity": "L2", "temporal_cluster": "0", "budget_bin": "mid", "roughness_bin": "low", "dominance_bin": "low", "coverage_bin": "low", "boundary_bin": "mid", "fragmentation_bin": "low", "retained_risk": 0.2, "ece": 0.1, "dice": 0.8},
            {"family": "random_missing_months", "severity": "L2", "temporal_cluster": "1", "budget_bin": "mid", "roughness_bin": "high", "dominance_bin": "high", "coverage_bin": "high", "boundary_bin": "high", "fragmentation_bin": "high", "retained_risk": 0.5, "ece": 0.3, "dice": 0.5},
            {"family": "random_missing_months", "severity": "L2", "temporal_cluster": "1", "budget_bin": "mid", "roughness_bin": "high", "dominance_bin": "high", "coverage_bin": "high", "boundary_bin": "high", "fragmentation_bin": "high", "retained_risk": 0.4, "ece": 0.2, "dice": 0.6},
        ]
        summary = summarize_group_results(records, rounds=50, confidence=0.95, seed=1)
        self.assertGreater(len(summary), 1)
        self.assertIn("group_count", summary.columns)
        worst = worst_group_summary(summary)
        self.assertIn("worst_group_risk_count", worst.columns)
        self.assertIn("worst_group_risk_ci_low", worst.columns)

    def test_table_semantics_are_preserved_for_degradation_and_risk_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            p02_dir = Path(tmp_dir) / "P02" / "seed01"
            p10_dir = Path(tmp_dir) / "P10" / "seed01"
            p05_dir = Path(tmp_dir) / "P05" / "seed01"
            (p02_dir / "tables").mkdir(parents=True)
            (p10_dir / "tables").mkdir(parents=True)
            (p05_dir / "tables").mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "family": "modality_drop",
                        "severity": "L1",
                        "ordinal_severity_family": False,
                        "severity_semantics": "non_ordinal_modality_drop",
                        "degradation_eval_scope": "family_severity_grid",
                        "degradation_eval_families": '["modality_drop"]',
                        "degradation_eval_severities": '{"modality_drop": ["L1", "L2"]}',
                        "dice": 0.5,
                        "iou": 0.4,
                        "ece": 0.2,
                        "nll": 0.3,
                        "brier": 0.2,
                        "feature_set": "core5",
                        "arm_role": "design_prior",
                    }
                ]
            ).to_csv(p02_dir / "tables" / "degradation_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "family": "modality_aware_radiometric_perturbation",
                        "severity": "L1",
                        "ordinal_severity_family": True,
                        "severity_semantics": "ordinal",
                        "degradation_eval_scope": "family_severity_grid",
                        "degradation_eval_families": '["modality_aware_radiometric_perturbation"]',
                        "degradation_eval_severities": '{"modality_aware_radiometric_perturbation": ["L1", "L2", "L3"]}',
                        "dice": 0.8,
                        "iou": 0.7,
                        "ece": 0.1,
                        "nll": 0.2,
                        "brier": 0.1,
                        "feature_set": "core5",
                        "arm_role": "design_prior",
                    }
                ]
            ).to_csv(p10_dir / "tables" / "degradation_summary.csv", index=False)
            frame = pd.DataFrame(
                [
                    {
                        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
                        "protocol_id": "P02",
                        "run_name": "P02__core5",
                        "arm_role": "design_prior",
                        "feature_set": "core5",
                        "feature_family_count": 5,
                        "tensor_channel_count": 5,
                        "seed": 1,
                        "config_path": "configs/protocols/P02.yaml",
                        "checkpoint_source": "outputs/checkpoints/P00__core5/seed01/checkpoint__best-dice.pt",
                        "run_dir": str(p02_dir),
                    },
                    {
                        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
                        "protocol_id": "P10",
                        "run_name": "P10__core5",
                        "arm_role": "design_prior",
                        "feature_set": "core5",
                        "feature_family_count": 5,
                        "tensor_channel_count": 5,
                        "seed": 1,
                        "config_path": "configs/protocols/P10.yaml",
                        "checkpoint_source": "outputs/checkpoints/P00__core5/seed01/checkpoint__best-dice.pt",
                        "run_dir": str(p10_dir),
                    },
                    {
                        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
                        "protocol_id": "P05",
                        "run_name": "P05__core5",
                        "arm_role": "design_prior",
                        "feature_set": "core5",
                        "feature_family_count": 5,
                        "tensor_channel_count": 5,
                        "seed": 1,
                        "config_path": "configs/protocols/P05.yaml",
                        "checkpoint_source": "outputs/checkpoints/P00__core5/seed01/checkpoint__best-dice.pt",
                        "calibrator_source": "outputs/checkpoints/P04__core5/seed01/calibrator__mode-global__family-random_missing_months__severity-L2.json",
                        "eval_family": "random_missing_months",
                        "eval_severity": "L2",
                        "paper_operating_points": [0.9, 0.8],
                        "fig5_semantics": "full risk-coverage curve",
                        "table4_semantics": "selected operating-point summary",
                        "coverage_0.90_risk": 0.2,
                        "coverage_0.80_risk": 0.15,
                        "run_dir": str(p05_dir),
                    },
                ]
            )
            pd.DataFrame(
                [
                    {
                        "protocol_id": "P05",
                        "run_name": "P05__core5",
                        "feature_set": "core5",
                        "arm_role": "design_prior",
                        "eval_family": "random_missing_months",
                        "eval_severity": "L2",
                        "coverage": 0.9,
                        "retained_risk": 0.2,
                    }
                ]
            ).to_csv(p05_dir / "tables" / "risk_coverage_summary.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "protocol_id": "P05",
                        "run_name": "P05__core5",
                        "feature_set": "core5",
                        "arm_role": "design_prior",
                        "eval_family": "random_missing_months",
                        "eval_severity": "L2",
                        "coverage": 0.9,
                        "retained_risk": 0.2,
                    }
                ]
            ).to_csv(p05_dir / "tables" / "risk_coverage_curve.csv", index=False)
            table3 = build_table3_degradation_benchmark(frame)
            table4 = build_table4_risk_control(frame)
            self.assertIn("severity_semantics", table3.columns)
            self.assertEqual(set(table3["protocol_id"]), {"P02", "P10"})
            self.assertFalse(bool(table3.iloc[0]["ordinal_severity_family"]))
            self.assertIn("fig5_semantics", table4.columns)
            self.assertIn("table4_semantics", table4.columns)
            self.assertIn("coverage_0.80_risk", table4.columns)


if __name__ == "__main__":
    unittest.main()
