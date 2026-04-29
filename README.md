# DegradeRisk-Seg

**DegradeRisk-Seg** is the paper code for risk-controlled semantic segmentation under degraded multi-modal observations.

`unified degradation protocol + risk-controlled prediction + endogenous heterogeneity analysis`

The current paper experiments use Plot-Rice v1.0 as the evaluation dataset, but the repository name and method framing are intentionally dataset-agnostic. Only manuscript-supported experiment entries, scripts, configuration files, tests, and lightweight metadata are retained. Full data archives, trained checkpoints, generated outputs, manuscript drafts, and local environment caches are intentionally excluded.

## Repository Status

- Research artifact for a degradation-aware segmentation pipeline.
- Author: Jiajun Chen, College of Earth Sciences, Jilin University. 中文：陈家骏，吉林大学地球科学学院。
- Default backbone: `Res U-Net`.
- Default design prior: `core5`.
- Stage-1 execution / validation arm: `core7`.
- Extended comparison arm: `full10`.
- License: MIT.
- Citation metadata: `CITATION.cff`. Add final paper title, DOI, and repository URL before tagging a release if available.

## Installation

Create a virtual environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Conda users can start from:

```bash
conda env create -f environment.yml
conda activate degraderisk-seg
```

## Data Setup

The code reads official Plot-Rice v1.0 zip packages directly; a full unpack is not required.

Expected local layout:

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

The default data configs point to `../PlotRice-V1.0`. If your data lives elsewhere, pass an override:

```bash
python scripts/train.py \
  --config configs/protocols/P00__core5__resunet__train-clean__eval-clean__seed01.yaml \
  --override data.source.root=/path/to/PlotRice-V1.0
```

Lightweight split and normalization metadata are versioned under `data/processed/`. Full data archives and generated arrays are ignored by `.gitignore`.

## Quick Run

Clean baseline:

```bash
python scripts/train.py --config configs/protocols/P00__core5__resunet__train-clean__eval-clean__seed01.yaml
```

Degradation benchmark:

```bash
python scripts/eval.py --config configs/protocols/P02__core5__resunet__train-clean__eval-degbench__seed01.yaml
```

Required calibration follow-up:

```bash
python scripts/calibrate.py --config configs/protocols/P04__core5__resunet__train-clean__eval-degbench__cal-global__seed01.yaml
```

Global selective baseline and descriptor-conditioned risk control:

```bash
python scripts/selective_predict.py --config configs/protocols/P05__core5__resunet__train-clean__eval-degbench__risk-global__seed01.yaml
python scripts/selective_predict.py --config configs/protocols/P06__core5__resunet__train-clean__eval-degbench__risk-descgrp__seed01.yaml
```

Integrated evidence protocols:

```bash
python scripts/calibration_decomposition.py --config configs/protocols/P08__core5__resunet__calibration-decomposition__seed01.yaml
python scripts/risk_score_comparison.py --config configs/protocols/P09__core5__resunet__risk-score-comparison__seed01.yaml
python scripts/eval.py --config configs/protocols/P10__core5__resunet__modality-aware-radiometric__seed01.yaml
```

The protocol YAML files are stored with `seed01` names as canonical run anchors. Additional manuscript seeds reuse the same protocol YAMLs:

```bash
python scripts/train.py --config configs/protocols/P00__core5__resunet__train-clean__eval-clean__seed01.yaml --override seed=2
python scripts/train.py --config configs/protocols/P00__core5__resunet__train-clean__eval-clean__seed01.yaml --override seed=3
```

See `docs/reproducibility.md` for the full paper-oriented run sequence.

## Protocol Semantics

- `P02` establishes degradation damage patterns.
- `P04` is the required follow-up for calibration collapse.
- `P05` is the global selective baseline.
- `P06` is the descriptor-conditioned extension over `P05`.
- `P08`, `P09`, and `P10` provide integrated evidence analyses used by the manuscript.

Feature-set semantics:

- `core5`: 5 feature families / 5 tensor channels.
- `core7`: 7 feature families / 7 tensor channels.
- `full10`: 10 feature families / 14 tensor channels.

Leakage boundary:

- `coverage`, `boundary complexity`, and `fragmentation` are analysis-only descriptors.
- They must not enter online calibration, thresholding, or descriptor grouping.

More details are in `docs/experiment_registry.md` and `docs/data_contract.md`.

## Artifacts

Runs are keyed by `experiment.run_name` and seed:

- Run outputs: `outputs/runs/<run_name>/seedXX/`.
- Checkpoints: `outputs/checkpoints/<run_name>/seedXX/checkpoint__best-dice.pt`.
- Calibrators: `outputs/checkpoints/<run_name>/seedXX/calibrator__mode-<mode>__family-<family>__severity-<severity>.json`.

Generated outputs are ignored by git. Publish trained checkpoints separately through a release asset or data archive if they are needed for reproduction.

## Tests

Run the lightweight test suite:

```bash
python -m pytest
```

The tests validate metric behavior, data-reader behavior, degradation semantics, protocol registration, and paper guards.

## Citation

If you use this repository, cite the associated paper and this software artifact. Add the final paper title, DOI, and repository URL to `CITATION.cff` before creating a formal release if available.

## License

This code is released under the MIT License. See `LICENSE`.
