from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _resolve_default(base_dir: Path, default: str) -> Path:
    candidate = (base_dir / default).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve default config: {default} from {base_dir}")


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    config = _read_yaml(path)
    defaults = config.pop("defaults", [])
    merged: dict[str, Any] = {}
    for default in defaults:
        if not isinstance(default, str):
            raise TypeError(f"Config defaults must be string paths, got {type(default)!r}")
        merged = merge_dicts(merged, load_config(_resolve_default(path.parent, default)))
    merged = merge_dicts(merged, config)
    merged.setdefault("meta", {})
    merged["meta"]["config_path"] = str(path)
    return merged


def _set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def apply_overrides(config: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return config
    updated = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override {item!r}, expected key=value")
        key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_nested(updated, key, value)
    return updated


def dump_config(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
    target.write_text(text, encoding="utf-8")

