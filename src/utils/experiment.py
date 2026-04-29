from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.config import dump_config
from src.utils.io import ensure_dir, write_json


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(*parts: str) -> Path:
    return project_root().joinpath(*parts)


def create_run_dir(config: dict[str, Any]) -> Path:
    experiment = config.get("experiment", {})
    exp_id = experiment.get("run_name", experiment.get("id", "unnamed_experiment"))
    seed = int(config.get("seed", 1))
    run_dir = resolve_path("outputs", "runs", exp_id, f"seed{seed:02d}")
    return ensure_dir(run_dir)


def bootstrap_run(config: dict[str, Any]) -> Path:
    run_dir = create_run_dir(config)
    ensure_dir(run_dir / "tables")
    ensure_dir(run_dir / "figures")
    dump_config(config, run_dir / "config.resolved.yaml")
    return run_dir


def save_metrics_summary(metrics: dict[str, Any], run_dir: Path) -> None:
    write_json(metrics, run_dir / "metrics_summary.json")
