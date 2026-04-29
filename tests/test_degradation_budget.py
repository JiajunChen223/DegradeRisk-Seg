from __future__ import annotations

import unittest

import numpy as np

from src.degradations.registry import build_degrader
from src.dataio.canonical_dataset import _apply_missing_value_strategy
from src.utils.config import load_config
from src.utils.experiment import project_root


class DegradationBudgetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(project_root() / "configs" / "protocols" / "P02__core5__resunet__train-clean__eval-degbench__seed01.yaml")
        self.x = np.ones((12, 5, 4, 4), dtype=np.float32)
        self.feature_families = ["DpRVIc", "sigma0_VH", "sigma0_VV", "NDVI", "LSWI"]
        self.feature_modalities = ["SAR", "SAR", "SAR", "OPT", "OPT"]

    def test_random_missing_months_budget(self) -> None:
        degrader = build_degrader(self.config, family_name="random_missing_months", severity_name="L2")
        _, obs_mask, meta = degrader(self.x, self.feature_families, self.feature_modalities, 1, "sample_a")
        self.assertEqual(obs_mask.shape, (12, 5))
        self.assertAlmostEqual(meta["retained_budget"], 8 / 12, places=3)

    def test_prefix_truncation_budget(self) -> None:
        degrader = build_degrader(self.config, family_name="prefix_truncation", severity_name="L3")
        _, obs_mask, meta = degrader(self.x, self.feature_families, self.feature_modalities, 1, "sample_b")
        self.assertAlmostEqual(meta["retained_budget"], 6 / 12, places=3)
        self.assertEqual(meta["longest_gap"], 6)

    def test_modality_aware_radiometric_perturbation_keeps_observation_mask(self) -> None:
        degrader = build_degrader(self.config, family_name="modality_aware_radiometric_perturbation", severity_name="L2")
        x_deg, obs_mask, meta = degrader(self.x, self.feature_families, self.feature_modalities, 1, "sample_c")
        self.assertAlmostEqual(float(obs_mask.mean()), 1.0, places=5)
        self.assertAlmostEqual(meta["retained_budget"], 1.0, places=5)
        self.assertFalse(np.allclose(x_deg, self.x))

    def test_missing_value_strategies_preserve_observation_mask_contract(self) -> None:
        x_reference = np.arange(4, dtype=np.float32).reshape(4, 1, 1, 1)
        x_degraded = x_reference.copy()
        obs_mask = np.ones((4, 1), dtype=np.float32)
        obs_mask[1:3, 0] = 0.0
        x_degraded[1:3, 0] = 0.0
        mean_filled, zero_after_norm = _apply_missing_value_strategy(x_reference, x_degraded, obs_mask, "temporal_mean")
        self.assertFalse(zero_after_norm)
        self.assertAlmostEqual(float(mean_filled[1, 0, 0, 0]), 1.5)
        nearest_filled, _ = _apply_missing_value_strategy(x_reference, x_degraded, obs_mask, "temporal_nearest")
        self.assertAlmostEqual(float(nearest_filled[1, 0, 0, 0]), 0.0)
        _, zero_after_norm = _apply_missing_value_strategy(x_reference, x_degraded, obs_mask, "normalized_zero")
        self.assertTrue(zero_after_norm)


if __name__ == "__main__":
    unittest.main()
