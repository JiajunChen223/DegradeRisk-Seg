from __future__ import annotations

import unittest

import numpy as np
import torch

from src.evaluators.calibration import calibration_bin_table, compute_binary_calibration_metrics, compute_calibration_decomposition
from src.evaluators.segmentation import batch_dice_iou
from src.risk_control.selective import risk_coverage_curve


class MetricsSmokeTest(unittest.TestCase):
    def test_segmentation_and_calibration_metrics(self) -> None:
        logits = torch.tensor([[[[5.0, -5.0], [5.0, -5.0]]]])
        target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
        seg = batch_dice_iou(logits, target)
        cal = compute_binary_calibration_metrics(logits, target.unsqueeze(1))
        self.assertGreater(seg["dice"], 0.99)
        self.assertGreater(seg["iou"], 0.99)
        self.assertLess(cal["ece"], 0.1)

    def test_calibration_decomposition_has_ece_components(self) -> None:
        logits = torch.tensor([[[[5.0, -5.0], [0.2, -0.2]]]])
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        rows = calibration_bin_table(logits, target, bins=5)
        decomp = compute_calibration_decomposition(logits, target, bins=5)
        self.assertEqual(len(rows), 5)
        self.assertAlmostEqual(decomp["ece"], decomp["overconfidence_ece"] + decomp["underconfidence_ece"], places=6)
        self.assertIn("high_confidence_error_rate", decomp)

    def test_risk_coverage_curve(self) -> None:
        scores = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
        losses = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
        curve = risk_coverage_curve(scores, losses)
        self.assertEqual(len(curve["coverage"]), 4)
        self.assertGreater(curve["aurc"], 0.0)


if __name__ == "__main__":
    unittest.main()
