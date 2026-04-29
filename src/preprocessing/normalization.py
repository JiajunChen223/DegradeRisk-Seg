from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.dataio.canonical_dataset import CanonicalPatchDataset
from src.utils.experiment import project_root
from src.utils.io import write_json


def compute_train_normalization(manifest_path: str | Path, split: str, feature_names: list[str]) -> dict[str, list[float]]:
    manifest = pd.read_csv(project_root() / manifest_path)
    rows = manifest.loc[manifest["split"] == split]
    channel_sums = None
    channel_sq_sums = None
    count = 0
    for row in rows.itertuples(index=False):
        npz = np.load(project_root() / getattr(row, "npz_path"), allow_pickle=True)
        x = npz["x"].astype(np.float32)
        stored_names = [str(item) for item in npz["feature_names"].tolist()]
        channel_ids = [stored_names.index(name) for name in feature_names]
        x = x[:, channel_ids]
        flat = x.transpose(1, 0, 2, 3).reshape(len(feature_names), -1)
        sums = flat.sum(axis=1)
        sq_sums = (flat**2).sum(axis=1)
        if channel_sums is None:
            channel_sums = sums
            channel_sq_sums = sq_sums
        else:
            channel_sums += sums
            channel_sq_sums += sq_sums
        count += flat.shape[1]
    if count == 0:
        raise RuntimeError("No training samples found for normalization")
    mean = channel_sums / count
    variance = np.maximum(channel_sq_sums / count - mean**2, 1e-6)
    stats = {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()}
    return stats


def save_normalization_stats(stats: dict[str, list[float]], path: str | Path) -> None:
    write_json(stats, project_root() / path)


def compute_dataset_normalization(dataset: CanonicalPatchDataset) -> dict[str, list[float]]:
    channel_sums = None
    channel_sq_sums = None
    count = 0
    for index in range(len(dataset)):
        if index == 0 or (index + 1) % 200 == 0 or (index + 1) == len(dataset):
            print(f"[normalization] processing {index + 1}/{len(dataset)}")
        sample = dataset[index]
        x = sample.x.numpy().astype(np.float32)
        flat = x.transpose(1, 0, 2, 3).reshape(len(dataset.feature_names), -1)
        sums = flat.sum(axis=1)
        sq_sums = (flat**2).sum(axis=1)
        if channel_sums is None:
            channel_sums = sums
            channel_sq_sums = sq_sums
        else:
            channel_sums += sums
            channel_sq_sums += sq_sums
        count += flat.shape[1]
    if count == 0:
        raise RuntimeError("No samples found for normalization")
    mean = channel_sums / count
    variance = np.maximum(channel_sq_sums / count - mean**2, 1e-6)
    return {"mean": mean.tolist(), "std": np.sqrt(variance).tolist()}
