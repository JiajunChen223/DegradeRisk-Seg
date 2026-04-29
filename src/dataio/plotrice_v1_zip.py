from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np

from src.utils.experiment import project_root


def _arr_key_rank(name: str) -> int:
    if not name.startswith("arr_"):
        return 10**9
    return int(name.split("_", 1)[1])


class PlotRiceV1ZipReader:
    def __init__(
        self,
        root: str | Path,
        feature_files: dict[str, str],
        label_file: str,
        num_months: int = 12,
        feature_band_indices: dict[str, int] | None = None,
        feature_bands_per_month: dict[str, int] | None = None,
    ) -> None:
        root_path = Path(root)
        if not root_path.is_absolute():
            root_path = (project_root() / root_path).resolve()
        self.root = root_path
        self.feature_files = feature_files
        self.label_file = label_file
        self.num_months = num_months
        self.feature_band_indices = feature_band_indices or {}
        self.feature_bands_per_month = feature_bands_per_month or {}
        self._zips: dict[str, zipfile.ZipFile] = {}

    def _get_zip(self, archive_name: str) -> zipfile.ZipFile:
        if archive_name not in self._zips:
            self._zips[archive_name] = zipfile.ZipFile(self.root / archive_name)
        return self._zips[archive_name]

    def _read_npz(self, archive_name: str, member_name: str) -> dict[str, np.ndarray]:
        zf = self._get_zip(archive_name)
        with zf.open(member_name) as handle:
            payload = handle.read()
        with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
            return {key: np.asarray(npz[key]) for key in npz.files}

    def _decode_features_from_npz(self, npz: dict[str, np.ndarray], feature_names: list[str]) -> dict[str, np.ndarray]:
        keys = sorted(npz.keys(), key=_arr_key_rank)
        stacked = np.stack([np.asarray(npz[key], dtype=np.float32) for key in keys], axis=0)
        decoded: dict[str, np.ndarray] = {}
        reshaped_cache: dict[int, np.ndarray] = {}
        for feature_name in feature_names:
            bands_per_month = int(self.feature_bands_per_month.get(feature_name, 1))
            if bands_per_month == 1:
                decoded[feature_name] = stacked
                continue
            expected = self.num_months * bands_per_month
            if stacked.shape[0] != expected:
                raise ValueError(
                    f"{feature_name} expects {expected} slices ({self.num_months}x{bands_per_month}), got {stacked.shape[0]}"
                )
            if bands_per_month not in reshaped_cache:
                reshaped_cache[bands_per_month] = stacked.reshape(self.num_months, bands_per_month, *stacked.shape[1:])
            band_index = int(self.feature_band_indices.get(feature_name, 0))
            decoded[feature_name] = reshaped_cache[bands_per_month][:, band_index]
        return decoded

    def read_feature(self, sample_id: str, feature_name: str) -> np.ndarray:
        archive_name = self.feature_files[feature_name]
        member_name = f"{sample_id}.npz"
        npz = self._read_npz(archive_name, member_name)
        return self._decode_features_from_npz(npz, [feature_name])[feature_name]

    def read_label(self, sample_id: str) -> np.ndarray:
        member_name = f"{sample_id}.npz"
        npz = self._read_npz(self.label_file, member_name)
        if "arr_0" not in npz:
            raise KeyError(f"Label archive for {sample_id} does not contain arr_0")
        return np.asarray(npz["arr_0"], dtype=np.float32)

    def read_sample(self, sample_id: str, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
        member_name = f"{sample_id}.npz"
        requested_by_archive: dict[str, list[str]] = {}
        for feature_name in feature_names:
            archive_name = self.feature_files[feature_name]
            requested_by_archive.setdefault(archive_name, []).append(feature_name)
        decoded: dict[str, np.ndarray] = {}
        for archive_name, archive_features in requested_by_archive.items():
            npz = self._read_npz(archive_name, member_name)
            decoded.update(self._decode_features_from_npz(npz, archive_features))
        features = [decoded[feature_name] for feature_name in feature_names]
        x = np.stack(features, axis=1)
        y = self.read_label(sample_id)
        return x, y

    def close(self) -> None:
        for zf in self._zips.values():
            zf.close()
        self._zips.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
