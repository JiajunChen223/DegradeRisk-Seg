# DegradeRisk-Seg

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-EE4C2C?logo=pytorch&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Task](https://img.shields.io/badge/task-risk--controlled%20segmentation-blue)](README.md)

**DegradeRisk-Seg** is a paper-oriented codebase for risk-controlled semantic segmentation under degraded multi-modal remote-sensing observations. It organizes degradation benchmarks, calibration analysis, selective prediction, and endogenous heterogeneity analysis in a reproducible experiment structure.

The reference experiments use Plot-Rice v1.0. The project name and method framing are dataset-agnostic: the degradation, calibration, and risk-control components can be adapted to segmentation datasets with similar temporal or multi-modal structure.

Author: **Jiajun Chen**, College of Earth Sciences, Jilin University.

## Highlights

| Theme | What This Repository Provides |
| --- | --- |
| Degraded observations | Random temporal missingness, structured temporal missingness, modality availability degradation, and modality-aware radiometric perturbation |
| Risk control | Global and descriptor-conditioned selective prediction |
| Calibration | Calibration collapse and calibration decomposition analyses under degraded observations |
| Heterogeneity | Endogenous heterogeneity analysis for retained-risk drivers |
| Reproducibility | Protocol registry, configs, tests, docs, and lightweight metadata |

## Repository Map

| Path | Purpose |
| --- | --- |
| `configs/` | Data, model, degradation, calibration, risk-control, and protocol configs |
| `data/` | Lightweight split and normalization metadata |
| `docs/` | Data contract, protocol registry, reproducibility guide, and release checklist |
| `scripts/` | Training, evaluation, calibration, selective prediction, and plotting entry points |
| `src/` | Core package code |
| `tests/` | Regression and paper-guard tests |

## Installation

Create a Python environment and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Conda:

```bash
conda env create -f environment.yml
conda activate degraderisk-seg
```

## Data Setup

The Plot-Rice v1.0 reader works directly with the official zip packages; a full unpack is not required.

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

Run the tests:

```bash
python -m pytest
```

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

See [docs/reproducibility.md](docs/reproducibility.md) for the full paper-oriented run sequence.

## Protocols

| ID | Role |
| --- | --- |
| `P00` | Clean baseline |
| `P02` | Degradation benchmark |
| `P04` | Calibration follow-up under degradation |
| `P05` | Global selective prediction baseline |
| `P06` | Descriptor-conditioned selective prediction |
| `P07` | Endogenous heterogeneity analysis |
| `P08` | Calibration decomposition |
| `P09` | Confidence-score and oracle selective-risk comparison |
| `P10` | Modality-aware radiometric perturbation |

Feature-set roles:

| Feature set | Role |
| --- | --- |
| `core5` | Compact multi-modal anchor, 5 feature families / 5 tensor channels |
| `core7` | Stage-1 execution and validation arm, 7 feature families / 7 tensor channels |
| `full10` | Extended comparison arm, 10 feature families / 14 tensor channels |

Details are documented in [docs/experiment_registry.md](docs/experiment_registry.md) and [docs/data_contract.md](docs/data_contract.md).

## Artifacts

Runs are keyed by `experiment.run_name` and seed:

```text
outputs/runs/<run_name>/seedXX/
outputs/checkpoints/<run_name>/seedXX/checkpoint__best-dice.pt
outputs/checkpoints/<run_name>/seedXX/calibrator__mode-<mode>__family-<family>__severity-<severity>.json
```

`outputs/` is ignored by Git. Archive or publish generated artifacts separately when releasing reproduced results.

## Citation

If you use this repository, please cite the associated paper and this software artifact. Citation metadata is provided in [CITATION.cff](CITATION.cff).

## License

This code is released under the MIT License. See [LICENSE](LICENSE).
