from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_plotrice_v1_manifest(
    dataset_root: str | Path,
    split_files: dict[str, str],
    manifest_path: str | Path,
    split_seed: int = 101,
    val_calib_ratio: float = 0.5,
) -> pd.DataFrame:
    root = Path(dataset_root)
    rows: list[dict[str, str]] = []
    for split_name, split_file in split_files.items():
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = root / split_path
        sample_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if split_name == "val" and "val_calib" not in split_files:
            rng = np.random.default_rng(split_seed)
            shuffled = sample_ids.copy()
            rng.shuffle(shuffled)
            cut = int(round(len(shuffled) * (1.0 - val_calib_ratio)))
            for sample_id in shuffled[:cut]:
                rows.append({"sample_id": sample_id, "split": "val"})
            for sample_id in shuffled[cut:]:
                rows.append({"sample_id": sample_id, "split": "val_calib"})
            continue
        for sample_id in sample_ids:
            rows.append({"sample_id": sample_id, "split": split_name})
    frame = pd.DataFrame(rows)
    manifest = Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(manifest, index=False)
    return frame
