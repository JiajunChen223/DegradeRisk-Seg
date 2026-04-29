from __future__ import annotations

import unittest

import numpy as np

from src.degradations.registry import build_degrader
from src.utils.config import load_config
from src.utils.experiment import project_root


class FamilySemanticsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(project_root() / "configs" / "protocols" / "P02__full10__resunet__train-clean__eval-degbench__seed01.yaml")
        self.family_labels = self.config["data"]["feature_families"]
        self.modalities = self.config["data"]["feature_modalities"]
        self.x = np.ones((12, len(self.family_labels), 4, 4), dtype=np.float32)

    def test_full10_exposes_true_family_granularity(self) -> None:
        self.assertEqual(len(set(self.family_labels)), 10)
        self.assertEqual(len(set(self.modalities)), 2)

    def test_irregular_missing_operates_on_family_ids(self) -> None:
        degrader = build_degrader(self.config, family_name="irregular_missing", severity_name="L2")
        _, obs_mask, meta = degrader(self.x, self.family_labels, self.modalities, 3, "sample_full10")
        self.assertEqual(meta["feature_family_count"], 10)
        self.assertEqual(len(meta["feature_family_labels"]), 10)
        true_colour_ids = [idx for idx, label in enumerate(self.family_labels) if label == "true_colour"]
        false_colour_ids = [idx for idx, label in enumerate(self.family_labels) if label == "false_colour"]
        self.assertEqual(len(true_colour_ids), 3)
        self.assertEqual(len(false_colour_ids), 3)
        self.assertTrue(np.all(obs_mask[:, true_colour_ids] == obs_mask[:, true_colour_ids[0:1]]))
        self.assertTrue(np.all(obs_mask[:, false_colour_ids] == obs_mask[:, false_colour_ids[0:1]]))

    def test_modality_drop_uses_modalities(self) -> None:
        degrader = build_degrader(self.config, family_name="modality_drop", severity_name="L1")
        _, obs_mask, _ = degrader(self.x, self.family_labels, self.modalities, 3, "sample_modality")
        opt_ids = [idx for idx, modality in enumerate(self.modalities) if modality == "OPT"]
        sar_ids = [idx for idx, modality in enumerate(self.modalities) if modality == "SAR"]
        self.assertTrue(np.all(obs_mask[:, opt_ids] == 0.0))
        self.assertTrue(np.all(obs_mask[:, sar_ids] == 1.0))

    def test_sar4_and_opt2_expose_distinct_families_and_modalities(self) -> None:
        sar4 = load_config(project_root() / "configs" / "data" / "sar4.yaml")["data"]
        opt2 = load_config(project_root() / "configs" / "data" / "opt2.yaml")["data"]
        self.assertEqual(sar4["feature_families"], ["DpRVIc", "sigma0_VH", "sigma0_VV", "gamma0_VH"])
        self.assertEqual(sar4["feature_modalities"], ["SAR", "SAR", "SAR", "SAR"])
        self.assertEqual(opt2["feature_families"], ["NDVI", "LSWI"])
        self.assertEqual(opt2["feature_modalities"], ["OPT", "OPT"])

    def test_unimodal_irregular_missing_operates_on_true_families(self) -> None:
        sar4_cfg = load_config(project_root() / "configs" / "data" / "sar4.yaml")["data"]
        x = np.ones((12, len(sar4_cfg["feature_families"]), 4, 4), dtype=np.float32)
        paper_cfg = load_config(project_root() / "configs" / "protocols" / "P02__core5__resunet__train-clean__eval-degbench__seed01.yaml")
        paper_cfg["data"] = sar4_cfg
        degrader = build_degrader(paper_cfg, family_name="irregular_missing", severity_name="L2")
        _, obs_mask, meta = degrader(x, sar4_cfg["feature_families"], sar4_cfg["feature_modalities"], 5, "sample_sar4")
        self.assertEqual(meta["feature_family_count"], 4)
        self.assertEqual(len(meta["feature_family_labels"]), 4)
        for idx in range(obs_mask.shape[1]):
            self.assertTrue(np.all(np.isin(obs_mask[:, idx], [0.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
