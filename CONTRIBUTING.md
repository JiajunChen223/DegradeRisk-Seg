# Contributing

Thank you for considering a contribution. This repository is a paper artifact, so changes should preserve the manuscript protocol semantics and reproducibility assumptions.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Pull Request Checklist

- Keep experiment IDs and run-name semantics aligned with `docs/experiment_registry.md`.
- Do not commit private data, full Plot-Rice zip packages, checkpoints, or generated `outputs/`.
- Add or update tests when changing protocol logic, metrics, data loading, degradation behavior, calibration, or selective prediction.
- Run `python -m pytest` before opening a pull request.

## Scope

Bug fixes, documentation improvements, reproducibility fixes, and small compatibility updates are welcome. New experimental branches should be clearly separated from the paper protocols unless they are part of an explicitly versioned follow-up.
