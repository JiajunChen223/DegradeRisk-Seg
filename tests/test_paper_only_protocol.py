from __future__ import annotations

import unittest

from src.utils.config import load_config
from src.utils.experiment import project_root


class PaperOnlyProtocolTest(unittest.TestCase):
    def test_integrated_protocols_are_registered(self) -> None:
        names = [
            "P08__core5__resunet__calibration-decomposition__seed01.yaml",
            "P09__core5__resunet__risk-score-comparison__seed01.yaml",
            "P10__core5__resunet__modality-aware-radiometric__seed01.yaml",
        ]
        for name in names:
            config = load_config(project_root() / "configs" / "protocols" / name)
            self.assertIn(config["experiment"]["protocol_id"], {"P08", "P09", "P10"})
            self.assertEqual(config["model"]["name"], "resunet")

    def test_p10_uses_modality_aware_radiometric_family(self) -> None:
        config = load_config(
            project_root()
            / "configs"
            / "protocols"
            / "P10__core5__resunet__modality-aware-radiometric__seed01.yaml"
        )
        self.assertEqual(config["experiment"]["protocol_id"], "P10")
        self.assertEqual(config["degradation"]["default_eval_family"], "modality_aware_radiometric_perturbation")
        self.assertEqual(config["degradation"]["eval_families"], ["modality_aware_radiometric_perturbation"])

    def test_p02_and_p10_degradation_grids_are_disjoint(self) -> None:
        p02 = load_config(
            project_root()
            / "configs"
            / "protocols"
            / "P02__core5__resunet__train-clean__eval-degbench__seed01.yaml"
        )
        p10 = load_config(
            project_root()
            / "configs"
            / "protocols"
            / "P10__core5__resunet__modality-aware-radiometric__seed01.yaml"
        )
        self.assertEqual(
            p02["degradation"]["eval_families"],
            [
                "random_missing_months",
                "block_missing_months",
                "prefix_truncation",
                "modality_drop",
                "irregular_missing",
            ],
        )
        p02_rows = sum(len(p02["degradation"]["families"][family]["severities"]) for family in p02["degradation"]["eval_families"])
        p10_rows = sum(len(p10["degradation"]["families"][family]["severities"]) for family in p10["degradation"]["eval_families"])
        self.assertEqual(p02_rows, 14)
        self.assertEqual(p10_rows, 3)


if __name__ == "__main__":
    unittest.main()
