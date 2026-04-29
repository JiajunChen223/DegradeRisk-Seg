# Plot-Rice Data Contract

## 1. Source Mode

The paper code uses the official Plot-Rice v1.0 zip-backed source:

- `source.type = plotrice_v1_zip`
- `source.root`
- `source.label_file`
- `source.split_files`
- `source.feature_files`

The loader reads feature and label zip packages directly. Full unpacking into a local NPZ cache is not part of the maintained paper path.

## 2. Split Manifest

`data/processed/split_manifests/plotrice_v1_official_split.csv` must contain:

- `sample_id`
- `split`, with values `train`, `val`, `val_calib`, or `test`

If the official validation split is provided without `val_calib`, the manifest builder deterministically splits validation into `val` and `val_calib` using `split_seed` and `official_val_calib_ratio`.

## 3. Data Config

Each `configs/data/*.yaml` entry must define:

- `feature_names`
- `feature_families`
- `feature_modalities`
- `num_months`
- `image_size`
- `positive_label`
- `manifest_path`
- `normalization.path`
- `source`

Feature-set configs are the only maintained way to switch between `core5`, `core7`, `full10`, `sar4`, and `opt2`.

## 4. Degradation Missing-Value Handling

The paper-only protocol keeps the fixed `raw_zero` behavior for synthetic missing observations. Alternative imputation sensitivity settings are outside this clean manuscript code path.

## 5. Leakage-Safe Descriptors

Allowed for test-time calibration, thresholding, and grouping:

- retained observation budget
- longest temporal gap
- modality-missing flags
- observability descriptors
- roughness or anomaly descriptors derived from observed inputs
- SAR-optical consistency or dominance descriptors
- train-fitted temporal cluster id

Analysis-only descriptors that must not enter online calibration, thresholding, or grouping:

- rice coverage
- boundary complexity
- fragmentation
