from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from src.dataio.plotrice_v1_zip import PlotRiceV1ZipReader


def _write_npz_bytes(path: Path, sample_id: str, arrays: list[np.ndarray]) -> None:
    tmp = path.parent / f"{sample_id}.npz"
    np.savez(tmp, *arrays)
    with zipfile.ZipFile(path, mode="w") as zf:
        zf.write(tmp, arcname=f"{sample_id}.npz")
    tmp.unlink()


def test_read_sample_reuses_shared_archive_payload(tmp_path: Path) -> None:
    sample_id = "sample_0001"
    feature_zip = tmp_path / "Feature-TrueColour.zip"
    label_zip = tmp_path / "Labels.zip"
    feature_arrays = [np.full((4, 4), fill_value=float(idx), dtype=np.float32) for idx in range(6)]
    label_array = np.ones((4, 4), dtype=np.uint8)
    _write_npz_bytes(feature_zip, sample_id, feature_arrays)
    _write_npz_bytes(label_zip, sample_id, [label_array])

    reader = PlotRiceV1ZipReader(
        root=tmp_path,
        feature_files={
            "true_colour_R": feature_zip.name,
            "true_colour_G": feature_zip.name,
            "true_colour_B": feature_zip.name,
        },
        label_file=label_zip.name,
        num_months=2,
        feature_band_indices={
            "true_colour_R": 0,
            "true_colour_G": 1,
            "true_colour_B": 2,
        },
        feature_bands_per_month={
            "true_colour_R": 3,
            "true_colour_G": 3,
            "true_colour_B": 3,
        },
    )

    original = reader._read_npz
    calls: list[tuple[str, str]] = []

    def counted(archive_name: str, member_name: str):
        calls.append((archive_name, member_name))
        return original(archive_name, member_name)

    reader._read_npz = counted  # type: ignore[method-assign]
    x, y = reader.read_sample(sample_id, ["true_colour_R", "true_colour_G", "true_colour_B"])
    reader.close()

    assert x.shape == (2, 3, 4, 4)
    assert y.shape == (4, 4)
    assert np.allclose(x[:, 0], np.stack([feature_arrays[0], feature_arrays[3]], axis=0))
    assert np.allclose(x[:, 1], np.stack([feature_arrays[1], feature_arrays[4]], axis=0))
    assert np.allclose(x[:, 2], np.stack([feature_arrays[2], feature_arrays[5]], axis=0))
    assert calls.count((feature_zip.name, f"{sample_id}.npz")) == 1
    assert calls.count((label_zip.name, f"{sample_id}.npz")) == 1
