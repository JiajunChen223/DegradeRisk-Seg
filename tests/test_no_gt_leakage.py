from __future__ import annotations

import unittest

import numpy as np

from src.descriptors.descriptor_extractor import extract_descriptors


class NoGTLeakageTest(unittest.TestCase):
    def test_analysis_only_keys_are_not_in_input_safe_grouping(self) -> None:
        x = np.random.rand(12, 5, 8, 8).astype(np.float32)
        y = (np.random.rand(8, 8) > 0.5).astype(np.float32)
        obs_mask = np.ones((12, 5), dtype=np.float32)
        payload = extract_descriptors(
            x=x,
            y=y,
            obs_mask=obs_mask,
            feature_names=["DpRVIc", "sigma0_VH", "sigma0_VV", "NDVI", "LSWI"],
            feature_families=["DpRVIc", "sigma0_VH", "sigma0_VV", "NDVI", "LSWI"],
            feature_modalities=["SAR", "SAR", "SAR", "OPT", "OPT"],
            degradation_meta={"retained_budget": 1.0, "longest_gap": 0},
        )
        input_safe = payload["input_safe"]
        analysis_only = payload["analysis_only"]
        self.assertIn("coverage", analysis_only)
        self.assertIn("boundary_complexity", analysis_only)
        self.assertNotIn("coverage", input_safe)
        self.assertNotIn("boundary_complexity", input_safe)
        self.assertIn("safe_group_key", input_safe)


if __name__ == "__main__":
    unittest.main()
