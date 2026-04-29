# Protocol Registry

This registry keeps the paper protocol IDs, run names, and artifact semantics aligned.

## Protocol Entries

- `P00`: clean baseline. Scripts: `train.py` (+ optional `eval.py` for clean reporting).
- `P01`: input-design definition. Source: `configs/data/*.yaml`.
  This slot documents CDI, SVI, ECI, OMI, and SCI semantics; it is not an optimization or ranking experiment.
- `P02`: degradation benchmark. Script: `eval.py`.
- `P03`: severity curves figure slot. Scripts: `eval.py + plot_main_figures.py`.
  This is a figure-generating analysis stage derived from degradation summaries, not a required standalone train/eval run config.
- `P04`: calibration collapse. Scripts: `calibrate.py` (+ optional `eval.py` / plotting).
- `P05`: selective prediction with the global baseline. Script: `selective_predict.py`.
- `P06`: descriptor-conditioned risk control. Scripts: `calibrate.py + selective_predict.py`.
- `P07`: endogenous heterogeneity analysis. Script: `heterogeneity_analysis.py`.
- `P08`: calibration decomposition. Script: `calibration_decomposition.py`.
- `P09`: confidence-score and oracle selective-risk comparison. Script: `risk_score_comparison.py`.
- `P10`: modality-aware radiometric perturbation. Script: `eval.py` with the degradation family restricted to `modality_aware_radiometric_perturbation`.
  This protocol supplies the observation-quality degradation rows used by the current manuscript protocol.

## Required Follow-Up Rule

- `P02` establishes degradation damage patterns.
- `P03` consumes degradation summaries as a figure slot; it does not introduce a separate model-training branch.
- `P04` is the required follow-up for calibration collapse.
- `P05` is the global selective baseline.
- `P06` is the descriptor-conditioned extension over `P05`, not a replacement for `P05`.
- `P08` / `P09` / `P10` consume existing checkpoints and calibrators where configured; they are archived as regular paper protocols in the same namespace.
- The current manuscript degradation aggregation uses P02 standard degradation rows plus P10 modality-aware radiometric perturbation rows.
- `configs/degradation/degbench.yaml` keeps the radiometric family definition available, while its default `eval_families` list is standard-only for P02; P10 explicitly selects `modality_aware_radiometric_perturbation`.

## Figure and Table Semantics

- `Fig.3` / `P03` shows ordinal severity curves for standard degradation families and a separate non-ordinal `modality_drop` panel.
- `Fig.5` is the full risk-coverage curve.
- `Table 4` is the selected operating-point summary aligned to the paper defaults (`0.90` and `0.80` coverage).

## Severity Semantics

- Degradation intensity levels use `L1` / `L2` / `L3` for low / medium / high severity.
- Clean evaluation uses `L0`.
- The `L*` convention avoids ambiguity with Sentinel-1 / Sentinel-2 shorthand.
- Standard severity ordering is defined by `ordinal_severity_family = family != "modality_drop"`.
- `modality_drop` is not treated as an ordinal severity family; it is reported as a separate non-ordinal degradation family in plots and tables.
- `modality_aware_radiometric_perturbation` is the observation-quality degradation family and is interpreted as ordinal radiometric perturbation severity.

## Arm Roles

- `core5`: `design_prior`
- `core7`: `stage1_validation_arm`
- `full10`: `comparison_arm`

The canonical wording is:

- `core5` = compact multi-modal anchor
- `core7` = stage-1 execution / validation arm
- `full10` = extended comparison arm

## Feature Set Counting

- `core5`: 5 feature families / 5 tensor channels
- `core7`: 7 feature families / 7 tensor channels
- `full10`: 10 feature families / 14 tensor channels
- `sar4`: 4 feature families / 4 tensor channels
- `opt2`: 2 feature families / 2 tensor channels

## Run and Artifact Mapping

Recommended run name format:

`<protocol-id>__<feature-set>__<arm-token>__<backbone>__<train-tag>__<eval-tag>__[extra-tag]`

Examples:

- `P00__core5__arm-designprior__resunet__train-clean__eval-clean`
- `P02__core7__arm-stage1val__resunet__train-clean__eval-degbench`
- `P05__full10__arm-compare__resunet__train-clean__eval-degbench__risk-global`

Artifacts are namespaced by run name and seed:

- `outputs/runs/<run_name>/seedXX/`
- `outputs/checkpoints/<run_name>/seedXX/checkpoint__best-dice.pt`
- `outputs/checkpoints/<run_name>/seedXX/calibrator__mode-<mode>__family-<family>__severity-<severity>.json`
