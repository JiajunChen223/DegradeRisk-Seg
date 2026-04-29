# DegradeRisk-Seg

**Risk-controlled semantic segmentation under degraded multi-modal observations.**

DegradeRisk-Seg accompanies a research pipeline for studying semantic segmentation when multi-modal remote-sensing observations are incomplete, perturbed, or unevenly available over time. The code provides a unified degradation benchmark, calibration analysis, selective prediction, and endogenous heterogeneity analysis in a reproducible experiment structure.

The reference experiments use Plot-Rice v1.0. The project name and method framing are dataset-agnostic: the degradation, calibration, and risk-control components are organized so they can be adapted to other segmentation datasets with similar temporal or multi-modal structure.

Author: **Jiajun Chen**, College of Earth Sciences, Jilin University.

## Highlights

- Unified degradation protocols for random temporal missingness, structured temporal missingness, modality availability degradation, and modality-aware radiometric perturbation.
- Risk-controlled selective prediction with both global and descriptor-conditioned operating points.
- Calibration collapse and calibration decomposition analyses under degraded observations.
- Endogenous heterogeneity analysis for understanding which observation conditions drive retained risk.
- Paper-ready protocol registry, configuration files, tests, and lightweight metadata for reproducible execution.

## Repository Layout

```text
configs/    Experiment, data, model, degradation, calibration, and risk-control configs
data/       Lightweight split and normalization metadata
docs/       Data contract, protocol registry, reproducibility guide, release checklist
scripts/    Training, evaluation, calibration, selective prediction, and plotting entry points
src/        Core package code
tests/      Regression and paper-guard tests
```

## Installation

Create a Python environment and install the project in editable mode:

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

The Plot-Rice v1.0 reader works directly with the official zip packages. A full unpack is not required.

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

The default configs use `../PlotRice-V1.0`. To use a different data location, pass an override:

```bash
python scripts/train.py \
  --config configs/protocols/P00__core5__resunet__train-clean__eval-clean__seed01.yaml \
  --override data.source.root=/path/to/PlotRice-V1.0
```

Split manifests and normalization statistics are stored under `data/processed/`. Generated arrays, checkpoints, and run outputs are kept outside version control through `.gitignore`.

## Quick Start

Train the clean baseline:

```bash
python scripts/train.py --config configs/protocols/P00__core5__resunet__train-clean__eval-clean__seed01.yaml
```

Evaluate the degradation benchmark:

```bash
python scripts/eval.py --config configs/protocols/P02__core5__resunet__train-clean__eval-degbench__seed01.yaml
```

Fit the global calibration follow-up:

```bash
python scripts/calibrate.py --config configs/protocols/P04__core5__resunet__train-clean__eval-degbench__cal-global__seed01.yaml
```

Compare global and descriptor-conditioned risk control:

```bash
python scripts/selective_predict.py --config configs/protocols/P05__core5__resunet__train-clean__eval-degbench__risk-global__seed01.yaml
python scripts/selective_predict.py --config configs/protocols/P06__core5__resunet__train-clean__eval-degbench__risk-descgrp__seed01.yaml
```

Run integrated evidence analyses:

```bash
python scripts/heterogeneity_analysis.py --config configs/protocols/P07__core5__resunet__train-clean__heterogeneity__seed01.yaml
python scripts/calibration_decomposition.py --config configs/protocols/P08__core5__resunet__calibration-decomposition__seed01.yaml
python scripts/risk_score_comparison.py --config configs/protocols/P09__core5__resunet__risk-score-comparison__seed01.yaml
python scripts/eval.py --config configs/protocols/P10__core5__resunet__modality-aware-radiometric__seed01.yaml
```

Repeat a protocol with additional seeds by overriding `seed`:

```bash
python scripts/train.py --config configs/protocols/P00__core5__resunet__train-clean__eval-clean__seed01.yaml --override seed=2
python scripts/train.py --config configs/protocols/P00__core5__resunet__train-clean__eval-clean__seed01.yaml --override seed=3
```

See `docs/reproducibility.md` for the full paper-oriented run sequence.

## Protocols

Core protocol IDs:

- `P00`: clean baseline.
- `P02`: degradation benchmark.
- `P04`: calibration follow-up under degradation.
- `P05`: global selective prediction baseline.
- `P06`: descriptor-conditioned selective prediction.
- `P07`: endogenous heterogeneity analysis.
- `P08`: calibration decomposition.
- `P09`: confidence-score and oracle selective-risk comparison.
- `P10`: modality-aware radiometric perturbation.

Feature-set roles:

- `core5`: compact multi-modal design prior, 5 feature families / 5 tensor channels.
- `core7`: stage-1 execution and validation arm, 7 feature families / 7 tensor channels.
- `full10`: extended comparison arm, 10 feature families / 14 tensor channels.

Descriptor boundary:

- Input-safe descriptors can be used for calibration, thresholding, and grouping.
- `coverage`, `boundary complexity`, and `fragmentation` are analysis-only descriptors.

Details are documented in `docs/experiment_registry.md` and `docs/data_contract.md`.

## Artifacts

Runs are keyed by `experiment.run_name` and seed:

```text
outputs/runs/<run_name>/seedXX/
outputs/checkpoints/<run_name>/seedXX/checkpoint__best-dice.pt
outputs/checkpoints/<run_name>/seedXX/calibrator__mode-<mode>__family-<family>__severity-<severity>.json
```

## Tests

Run the test suite:

```bash
python -m pytest
```

The tests cover metric behavior, data-reader behavior, degradation semantics, protocol registration, and paper-guard invariants.

## Citation

If you use this repository, please cite the associated paper and this software artifact. Citation metadata is provided in `CITATION.cff`.

## License

This code is released under the MIT License. See `LICENSE`.
