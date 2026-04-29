from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.preprocessing.canonical_builder import build_plotrice_v1_manifest


class PlotRiceManifestTest(unittest.TestCase):
    def test_val_is_split_into_val_and_val_calib(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "train.txt").write_text("A-1\nA-2\n", encoding="utf-8")
            (root / "val.txt").write_text("B-1\nB-2\nB-3\nB-4\n", encoding="utf-8")
            (root / "test.txt").write_text("C-1\n", encoding="utf-8")
            manifest_path = root / "manifest.csv"
            frame = build_plotrice_v1_manifest(
                dataset_root=root,
                split_files={"train": "train.txt", "val": "val.txt", "test": "test.txt"},
                manifest_path=manifest_path,
                split_seed=123,
                val_calib_ratio=0.5,
            )
            saved = pd.read_csv(manifest_path)
            self.assertEqual(len(frame), 7)
            self.assertEqual(len(saved), 7)
            counts = saved["split"].value_counts().to_dict()
            self.assertEqual(counts["train"], 2)
            self.assertEqual(counts["test"], 1)
            self.assertEqual(counts["val"], 2)
            self.assertEqual(counts["val_calib"], 2)


if __name__ == "__main__":
    unittest.main()

