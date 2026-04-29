from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.dataio.plotrice_v1_zip import PlotRiceV1ZipReader
from src.utils.experiment import project_root


@dataclass
class PatchSample:
    sample_id: str
    x: torch.Tensor
    y: torch.Tensor
    obs_mask: torch.Tensor
    degradation_meta: dict[str, Any]
    feature_names: list[str]
    feature_families: list[str]
    feature_modalities: list[str]


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return project_root() / path


def _fill_temporal_mean(x_reference: np.ndarray, x_degraded: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    filled = x_degraded.copy()
    for channel in range(filled.shape[1]):
        observed = np.flatnonzero(obs_mask[:, channel] > 0.0)
        missing = np.flatnonzero(obs_mask[:, channel] <= 0.0)
        if not missing.size:
            continue
        if not observed.size:
            filled[missing, channel] = 0.0
            continue
        filled[missing, channel] = x_reference[observed, channel].mean(axis=0)
    return filled


def _fill_temporal_nearest(x_reference: np.ndarray, x_degraded: np.ndarray, obs_mask: np.ndarray) -> np.ndarray:
    filled = x_degraded.copy()
    for channel in range(filled.shape[1]):
        observed = np.flatnonzero(obs_mask[:, channel] > 0.0)
        missing = np.flatnonzero(obs_mask[:, channel] <= 0.0)
        if not missing.size:
            continue
        if not observed.size:
            filled[missing, channel] = 0.0
            continue
        for month in missing.tolist():
            nearest = int(observed[np.argmin(np.abs(observed - month))])
            filled[month, channel] = x_reference[nearest, channel]
    return filled


def _apply_missing_value_strategy(
    x_reference: np.ndarray,
    x_degraded: np.ndarray,
    obs_mask: np.ndarray,
    strategy: str,
) -> tuple[np.ndarray, bool]:
    strategy = strategy.lower()
    if strategy in {"raw_zero", "zero", "zero_fill"}:
        return x_degraded, False
    if strategy in {"normalized_zero", "post_norm_zero"}:
        return x_degraded, True
    if strategy == "temporal_mean":
        return _fill_temporal_mean(x_reference, x_degraded, obs_mask), False
    if strategy == "temporal_nearest":
        return _fill_temporal_nearest(x_reference, x_degraded, obs_mask), False
    raise KeyError(f"Unknown missing value strategy: {strategy}")


class CanonicalPatchDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        feature_names: list[str],
        feature_families: list[str],
        feature_modalities: list[str] | None = None,
        source_cfg: dict[str, Any] | None = None,
        normalization_stats: dict[str, list[float]] | None = None,
        degrader: Callable[[np.ndarray, list[str], list[str], int, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]] | None = None,
        missing_value_strategy: str = "raw_zero",
        seed: int = 1,
    ) -> None:
        manifest = pd.read_csv(_resolve_path(str(manifest_path)))
        split_rows = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        self.sample_ids = split_rows["sample_id"].astype(str).tolist()
        self.npz_paths = split_rows["npz_path"].astype(str).tolist() if "npz_path" in split_rows.columns else None
        self.feature_names = feature_names
        self.feature_families = feature_families
        self.feature_modalities = feature_modalities or feature_families
        self.source_cfg = source_cfg or {}
        self.normalization_stats = normalization_stats
        self.degrader = degrader
        self.missing_value_strategy = missing_value_strategy
        self.seed = seed
        self.zip_reader: PlotRiceV1ZipReader | None = None
        if self.source_cfg.get("type") == "plotrice_v1_zip":
            self.zip_reader = PlotRiceV1ZipReader(
                root=self.source_cfg["root"],
                feature_files=self.source_cfg["feature_files"],
                label_file=self.source_cfg["label_file"],
                num_months=int(self.source_cfg.get("num_months", 12)),
                feature_band_indices=self.source_cfg.get("feature_band_indices"),
                feature_bands_per_month=self.source_cfg.get("feature_bands_per_month"),
            )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_npz(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        if self.zip_reader is not None:
            return self.zip_reader.read_sample(self.sample_ids[index], self.feature_names)
        if self.npz_paths is None:
            raise KeyError("Manifest is missing npz_path for canonical dataset loading")
        npz = np.load(_resolve_path(self.npz_paths[index]), allow_pickle=True)
        x = npz["x"].astype(np.float32)
        y = npz["y"].astype(np.float32)
        if "feature_names" in npz:
            stored_names = [str(item) for item in npz["feature_names"].tolist()]
            index_map = [stored_names.index(name) for name in self.feature_names]
            x = x[:, index_map]
        return x, y

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if not self.normalization_stats:
            return x
        mean = np.asarray(self.normalization_stats["mean"], dtype=np.float32).reshape(1, -1, 1, 1)
        std = np.asarray(self.normalization_stats["std"], dtype=np.float32).reshape(1, -1, 1, 1)
        return (x - mean) / np.clip(std, 1e-6, None)

    def __getitem__(self, index: int) -> PatchSample:
        x, y = self._load_npz(index)
        x_reference = x.copy()
        sample_id = self.sample_ids[index]
        if self.degrader is None:
            obs_mask = np.ones((x.shape[0], x.shape[1]), dtype=np.float32)
            meta = {
                "family": "clean",
                "severity": "L0",
                "retained_budget": 1.0,
                "longest_gap": 0,
            }
        else:
            x, obs_mask, meta = self.degrader(x, self.feature_families, self.feature_modalities, self.seed, sample_id)
        x, zero_after_normalization = _apply_missing_value_strategy(x_reference, x, obs_mask, self.missing_value_strategy)
        x = self._normalize(x)
        if zero_after_normalization:
            x = np.where(obs_mask[:, :, None, None] > 0.0, x, 0.0)
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y).float()
        obs_mask_tensor = torch.from_numpy(obs_mask).float()
        return PatchSample(
            sample_id=sample_id,
            x=x_tensor,
            y=y_tensor,
            obs_mask=obs_mask_tensor,
            degradation_meta=meta,
            feature_names=self.feature_names,
            feature_families=self.feature_families,
            feature_modalities=self.feature_modalities,
        )


def collate_patch_samples(batch: list[PatchSample]) -> dict[str, Any]:
    return {
        "sample_id": [item.sample_id for item in batch],
        "x": torch.stack([item.x for item in batch], dim=0),
        "y": torch.stack([item.y for item in batch], dim=0),
        "obs_mask": torch.stack([item.obs_mask for item in batch], dim=0),
        "degradation_meta": [item.degradation_meta for item in batch],
        "feature_names": batch[0].feature_names if batch else [],
        "feature_families": batch[0].feature_families if batch else [],
        "feature_modalities": batch[0].feature_modalities if batch else [],
    }
