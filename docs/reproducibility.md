# Reproducibility Guide

This guide describes the paper-oriented execution path for the DegradeRisk-Seg code release.

## 1. Environment

Recommended setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Alternative Conda setup:

```bash
conda env create -f environment.yml
conda activate degraderisk-seg
```

Verify the lightweight tests:

```bash
python -m pytest
```

## 2. Data Placement

Place official Plot-Rice v1.0 zip packages next to the repository:

```text
<workspace>/
  DegradeRisk-Seg/
  PlotRice-V1.0/
    Labels.zip
    Feature-DpRVIc.zip
    Feature-VHSigma0.zip
    Feature-VVSigma0.zip
    Feature-NDVI.zip
    Feature-LSWI.zip
    ...
```

The maintained paper path reads zip files directly through `source.type = plotrice_v1_zip`. The default configs use `data.source.root = ../PlotRice-V1.0`.

To use a different location:

```bash
--override data.source.root=/path/to/PlotRice-V1.0
```

## 3. Canonical Metadata

The repository includes lightweight split and normalization metadata:

- `data/processed/split_manifests/plotrice_v1_official_split.csv`
- `data/processed/canonical/normalization_core5_plotrice_v1.json`
- `data/processed/canonical/normalization_core7_plotrice_v1.json`
- `data/processed/canonical/normalization_full10_plotrice_v1.json`

If the manifest or normalization files are missing, the scripts can rebuild them from the official zip-backed source.

## 4. Paper Run Sequence

Run each feature-set arm as needed: `core5`, `core7`, and `full10`. The examples below use `core5`.

Train the clean baseline:

```bash
python scripts/train.py --config configs/protocols/P00__core5__resunet__train-clean__eval-clean__seed01.yaml
```

Evaluate standard degradation families:

```bash
python scripts/eval.py --config configs/protocols/P02__core5__resunet__train-clean__eval-degbench__seed01.yaml
```

Fit global calibration after degradation:

```bash
python scripts/calibrate.py --config configs/protocols/P04__core5__resunet__train-clean__eval-degbench__cal-global__seed01.yaml
```

Run the global selective baseline:

```bash
python scripts/selective_predict.py --config configs/protocols/P05__core5__resunet__train-clean__eval-degbench__risk-global__seed01.yaml
```

Run descriptor-conditioned selective prediction:

```bash
python scripts/selective_predict.py --config configs/protocols/P06__core5__resunet__train-clean__eval-degbench__risk-descgrp__seed01.yaml
```

Run heterogeneity and integrated evidence analyses:

```bash
python scripts/heterogeneity_analysis.py --config configs/protocols/P07__core5__resunet__train-clean__heterogeneity__seed01.yaml
python scripts/calibration_decomposition.py --config configs/protocols/P08__core5__resunet__calibration-decomposition__seed01.yaml
python scripts/risk_score_comparison.py --config configs/protocols/P09__core5__resunet__risk-score-comparison__seed01.yaml
python scripts/eval.py --config configs/protocols/P10__core5__resunet__modality-aware-radiometric__seed01.yaml
```

Repeat with `--override seed=2` and `--override seed=3` for the additional manuscript seeds.

## 5. Artifact Locations

Outputs are namespaced by run name and seed:

- `outputs/runs/<run_name>/seedXX/`
- `outputs/checkpoints/<run_name>/seedXX/checkpoint__best-dice.pt`
- `outputs/checkpoints/<run_name>/seedXX/calibrator__mode-<mode>__family-<family>__severity-<severity>.json`

`outputs/` is ignored by git. Archive or publish outputs separately when releasing reproduced results.

## 6. Reproducibility Notes

- Protocol IDs and run-name semantics are documented in `docs/experiment_registry.md`.
- Data schema and leakage-safe descriptor boundaries are documented in `docs/data_contract.md`.
- The code sets the configured seed through `src.utils.io.set_seed`.
- GPU kernels and PyTorch versions can still introduce small numerical differences across systems.
