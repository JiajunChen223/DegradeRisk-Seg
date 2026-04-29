from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts._bootstrap import ROOT
from src.dataio.canonical_dataset import CanonicalPatchDataset, collate_patch_samples
from src.descriptors.descriptor_extractor import extract_descriptors
from src.evaluators.segmentation import flatten_time_channels, per_sample_dice_loss
from src.models.factory import build_model
from src.preprocessing.canonical_builder import build_plotrice_v1_manifest
from src.preprocessing.normalization import compute_dataset_normalization, compute_train_normalization, save_normalization_stats
from src.utils.config import apply_overrides, load_config
from src.utils.experiment import bootstrap_run, project_root
from src.utils.io import read_json, set_seed


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--override", action="append", default=[], help="Dotlist override, e.g. train.epochs=10")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-normalization", action="store_true", help="Only prepare manifest/source metadata and skip normalization stats")
    return parser.parse_args()


def load_experiment(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    config = load_config(ROOT / args.config if not Path(args.config).is_absolute() else args.config)
    config = apply_overrides(config, args.override)
    data_cfg = config.get("data", {})
    config.setdefault("meta", {})
    config["meta"]["active_data_source_type"] = data_cfg.get("source", {}).get("type", "canonical_npz")
    config["meta"]["active_manifest_path"] = data_cfg.get("manifest_path")
    set_seed(int(config.get("seed", 1)))
    run_dir = bootstrap_run(config)
    return config, run_dir


def experiment_run_name(config: dict[str, Any]) -> str:
    experiment = config.get("experiment", {})
    return str(experiment.get("run_name", experiment.get("id", "unnamed_experiment")))


def _stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _artifact_seed_dir(run_name: str, seed: int) -> Path:
    return project_root() / "outputs" / "checkpoints" / run_name / f"seed{int(seed):02d}"


def runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("runtime", {}))


def use_channels_last(config: dict[str, Any]) -> bool:
    return bool(runtime_config(config).get("channels_last", False))


def configure_runtime(config: dict[str, Any], device: torch.device) -> None:
    if device.type != "cuda":
        return
    runtime_cfg = runtime_config(config)
    if "allow_tf32" in runtime_cfg:
        allow_tf32 = bool(runtime_cfg["allow_tf32"])
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    if "cudnn_benchmark" in runtime_cfg:
        torch.backends.cudnn.benchmark = bool(runtime_cfg["cudnn_benchmark"])
    matmul_precision = runtime_cfg.get("float32_matmul_precision")
    if matmul_precision:
        torch.set_float32_matmul_precision(str(matmul_precision))


def prepare_model_inputs(batch_x: torch.Tensor, device: torch.device, config: dict[str, Any]) -> torch.Tensor:
    x = batch_x.to(device, non_blocking=True)
    inputs = flatten_time_channels(x)
    if device.type == "cuda" and use_channels_last(config):
        inputs = inputs.contiguous(memory_format=torch.channels_last)
    return inputs


def checkpoint_artifact_path(config: dict[str, Any], run_name: str | None = None, seed: int | None = None, kind: str = "best-dice") -> Path:
    active_run_name = run_name or experiment_run_name(config)
    active_seed = int(config.get("seed", 1) if seed is None else seed)
    return _artifact_seed_dir(active_run_name, active_seed) / f"checkpoint__{kind}.pt"


def calibrator_artifact_path(
    config: dict[str, Any],
    run_name: str | None = None,
    seed: int | None = None,
    mode: str | None = None,
    family: str | None = None,
    severity: str | None = None,
) -> Path:
    active_run_name = run_name or experiment_run_name(config)
    active_seed = int(config.get("seed", 1) if seed is None else seed)
    calib_mode = str(mode or config.get("calibration", {}).get("mode", "global"))
    eval_family = str(family or config.get("degradation", {}).get("default_eval_family", "clean"))
    eval_severity = str(severity or config.get("degradation", {}).get("severity", "L0"))
    return _artifact_seed_dir(active_run_name, active_seed) / f"calibrator__mode-{calib_mode}__family-{eval_family}__severity-{eval_severity}.json"


def source_checkpoint_ref(config: dict[str, Any]) -> tuple[str, int]:
    artifacts = config.get("artifacts", {})
    run_name = str(artifacts.get("source_checkpoint_run_name", experiment_run_name(config)))
    seed = int(artifacts.get("source_checkpoint_seed", config.get("seed", 1)))
    return run_name, seed


def source_calibrator_ref(config: dict[str, Any]) -> tuple[str, int]:
    artifacts = config.get("artifacts", {})
    run_name = str(artifacts.get("source_calibrator_run_name", experiment_run_name(config)))
    seed = int(artifacts.get("source_calibrator_seed", config.get("seed", 1)))
    return run_name, seed


def resolve_checkpoint_path(config: dict[str, Any], path: Path | None = None, kind: str = "best-dice") -> Path:
    if path is not None:
        return path
    run_name, seed = source_checkpoint_ref(config)
    canonical = checkpoint_artifact_path(config, run_name=run_name, seed=seed, kind=kind)
    if canonical.exists():
        return canonical
    raise FileNotFoundError(f"Could not resolve checkpoint for run={run_name}, seed={seed}: {canonical}")


def resolve_calibrator_path(config: dict[str, Any], path: Path | None = None) -> Path:
    if path is not None:
        return path
    run_name, seed = source_calibrator_ref(config)
    canonical = calibrator_artifact_path(config, run_name=run_name, seed=seed)
    if canonical.exists():
        return canonical
    raise FileNotFoundError(f"Could not resolve calibrator for run={run_name}, seed={seed}: {canonical}")


def summary_provenance(
    config: dict[str, Any],
    checkpoint_source: Path | None = None,
    calibrator_source: Path | None = None,
) -> dict[str, Any]:
    experiment = config.get("experiment", {})
    data_cfg = config.get("data", {})
    feature_families = list(data_cfg.get("feature_families", []))
    feature_modalities = list(data_cfg.get("feature_modalities", feature_families))
    runtime_cfg = runtime_config(config)
    return {
        "summary_schema_version": "paper_v2",
        "protocol_id": experiment.get("protocol_id"),
        "experiment_id": experiment.get("id"),
        "run_name": experiment_run_name(config),
        "arm_role": experiment.get("arm_role"),
        "arm_label": experiment.get("arm_label"),
        "design_prior_ref": experiment.get("design_prior_ref"),
        "required_next": list(experiment.get("required_next", [])),
        "feature_set": data_cfg.get("name"),
        "feature_set_label": data_cfg.get("feature_set_label", data_cfg.get("name")),
        "feature_family_count": int(data_cfg.get("feature_family_count", len(data_cfg.get("feature_names", [])))),
        "tensor_channel_count": int(data_cfg.get("tensor_channel_count", len(data_cfg.get("feature_names", [])))),
        "feature_family_labels": _stable_unique(feature_families),
        "feature_modality_labels": _stable_unique(feature_modalities),
        "seed": int(config.get("seed", 1)),
        "config_path": config.get("meta", {}).get("config_path"),
        "data_source_type": config.get("meta", {}).get("active_data_source_type", data_cfg.get("source", {}).get("type")),
        "manifest_path": config.get("meta", {}).get("active_manifest_path", data_cfg.get("manifest_path")),
        "runtime_allow_tf32": runtime_cfg.get("allow_tf32"),
        "runtime_cudnn_benchmark": runtime_cfg.get("cudnn_benchmark"),
        "runtime_channels_last": runtime_cfg.get("channels_last"),
        "runtime_float32_matmul_precision": runtime_cfg.get("float32_matmul_precision"),
        "checkpoint_source": str(checkpoint_source) if checkpoint_source is not None else None,
        "calibrator_source": str(calibrator_source) if calibrator_source is not None else None,
    }


def maybe_prepare_data(config: dict[str, Any], prepare_normalization: bool = True) -> None:
    manifest = project_root() / config["data"]["manifest_path"]
    source_cfg = config["data"].get("source", {})
    source_type = source_cfg.get("type")
    if source_type == "plotrice_v1_zip":
        if not manifest.exists():
            build_plotrice_v1_manifest(
                dataset_root=source_cfg["root"],
                split_files=source_cfg["split_files"],
                manifest_path=manifest,
                split_seed=int(config["split"].get("split_seed", 101)),
                val_calib_ratio=float(config["split"].get("official_val_calib_ratio", 0.5)),
            )
    else:
        if not manifest.exists():
            raise FileNotFoundError(f"Canonical manifest not found: {manifest}")
    if not prepare_normalization:
        return
    norm_path = project_root() / config["data"]["normalization"]["path"]
    if not norm_path.exists():
        if source_type == "plotrice_v1_zip":
            dataset = CanonicalPatchDataset(
                manifest_path=config["data"]["manifest_path"],
                split=config["split"]["train"],
                feature_names=config["data"]["feature_names"],
                feature_families=config["data"]["feature_families"],
                feature_modalities=config["data"].get("feature_modalities", config["data"]["feature_families"]),
                source_cfg=source_cfg,
                normalization_stats=None,
                degrader=None,
                seed=int(config.get("seed", 1)),
            )
            stats = compute_dataset_normalization(dataset)
        else:
            stats = compute_train_normalization(
                manifest_path=config["data"]["manifest_path"],
                split=config["split"]["train"],
                feature_names=config["data"]["feature_names"],
            )
        save_normalization_stats(stats, config["data"]["normalization"]["path"])


def load_normalization_stats(config: dict[str, Any]) -> dict[str, list[float]]:
    return read_json(project_root() / config["data"]["normalization"]["path"])


def resolve_pos_weight_value(config: dict[str, Any], loader: DataLoader | None = None) -> float:
    train_cfg = config.get("train", {})
    mode = str(train_cfg.get("pos_weight_mode", "fixed"))
    if mode != "estimate_train":
        return float(train_cfg.get("pos_weight", 1.0))
    if loader is None:
        raise ValueError("pos_weight_mode=estimate_train requires a loader")
    positive = 0.0
    total = 0.0
    for batch in loader:
        y = batch["y"]
        positive += float(y.sum().item())
        total += float(y.numel())
    negative = max(total - positive, 0.0)
    if positive <= 0.0:
        return 1.0
    return float(negative / positive)


def make_loader(
    config: dict[str, Any],
    split_name: str,
    batch_size: int | None = None,
    degrader=None,
    shuffle: bool = False,
) -> DataLoader:
    stats = load_normalization_stats(config)
    train_cfg = config.get("train", {})
    num_workers = int(train_cfg.get("num_workers", 0))
    pin_memory = bool(train_cfg.get("pin_memory", torch.cuda.is_available()))
    persistent_workers = bool(train_cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(train_cfg.get("prefetch_factor", 2))
    dataset = CanonicalPatchDataset(
        manifest_path=config["data"]["manifest_path"],
        split=split_name,
        feature_names=config["data"]["feature_names"],
        feature_families=config["data"]["feature_families"],
        feature_modalities=config["data"].get("feature_modalities", config["data"]["feature_families"]),
        source_cfg=config["data"].get("source"),
        normalization_stats=stats,
        degrader=degrader,
        missing_value_strategy=str(config.get("degradation", {}).get("missing_value_strategy", "raw_zero")),
        seed=int(config.get("seed", 1)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size or int(train_cfg["batch_size"]),
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=collate_patch_samples,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def build_model_and_device(config: dict[str, Any], device_name: str) -> tuple[torch.nn.Module, torch.device]:
    in_channels = int(config["data"]["num_months"]) * len(config["data"]["feature_names"])
    model = build_model(config["model"], in_channels=in_channels)
    device = torch.device(device_name)
    configure_runtime(config, device)
    model.to(device)
    if device.type == "cuda" and use_channels_last(config):
        model.to(memory_format=torch.channels_last)
    return model, device


def save_checkpoint(model: torch.nn.Module, config: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, target)


def load_checkpoint(model: torch.nn.Module, config: dict[str, Any], path: Path | None = None) -> Path:
    ckpt_path = resolve_checkpoint_path(config, path=path)
    payload = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(payload["model_state"])
    return ckpt_path


def save_calibrator_state(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_calibrator_from_config(config: dict[str, Any]):
    from src.calibration.temperature_scaling import GroupedTemperatureScaler, TemperatureScaler

    path = resolve_calibrator_path(config)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("type") == "descriptor_group":
        return GroupedTemperatureScaler.from_state_dict(payload), path
    if "temperature" in payload:
        return TemperatureScaler.from_state_dict(payload), path
    raise ValueError(f"Unsupported calibrator payload: {path}")


def run_model_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any] | None = None,
    calibrator=None,
    groupwise: bool = False,
    group_keys: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    active_config = config or {}
    all_logits = []
    all_targets = []
    all_sample_ids: list[str] = []
    all_metas: list[dict[str, Any]] = []
    all_descriptors: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"]
            y = batch["y"].to(device, non_blocking=True)
            inputs = prepare_model_inputs(x, device, active_config)
            logits = model(inputs)
            descriptors = []
            groups = []
            for idx in range(x.shape[0]):
                descriptor = extract_descriptors(
                    x=x[idx].detach().cpu().numpy(),
                    y=y[idx].detach().cpu().numpy(),
                    obs_mask=batch["obs_mask"][idx].detach().cpu().numpy(),
                    feature_names=batch["feature_names"],
                    feature_families=batch["feature_families"],
                    feature_modalities=batch["feature_modalities"],
                    degradation_meta=batch["degradation_meta"][idx],
                )
                descriptors.append(descriptor)
                groups.append(compose_descriptor_group(descriptor, group_keys=group_keys))
            if calibrator is not None:
                from src.calibration.temperature_scaling import apply_calibration

                logits = apply_calibration(logits, calibrator, groups if groupwise else None)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu().unsqueeze(1))
            all_sample_ids.extend(batch["sample_id"])
            all_metas.extend(batch["degradation_meta"])
            all_descriptors.extend(descriptors)
    logits_tensor = torch.cat(all_logits, dim=0)
    targets_tensor = torch.cat(all_targets, dim=0)
    losses = per_sample_dice_loss(logits_tensor, targets_tensor)
    return {
        "logits": logits_tensor,
        "targets": targets_tensor,
        "sample_ids": all_sample_ids,
        "metas": all_metas,
        "descriptors": all_descriptors,
        "dice_losses": losses,
    }


def compose_descriptor_group(descriptor: dict[str, Any], group_keys: list[str] | None = None) -> str:
    input_safe = descriptor["input_safe"]
    keys = list(group_keys or ["budget_bin", "longest_gap_bin", "phenology_shape_bin", "cross_modal_consistency_bin"])
    return "|".join(str(input_safe.get(key, "na")) for key in keys)


def descriptor_groups(descriptors: Iterable[dict[str, Any]], group_keys: list[str] | None = None) -> list[str]:
    return [compose_descriptor_group(item, group_keys=group_keys) for item in descriptors]


def resolve_degradation_grid(config: dict[str, Any], analysis_cfg: dict[str, Any] | None = None) -> tuple[list[str], dict[str, list[str]]]:
    analysis_cfg = analysis_cfg or {}
    degradation_cfg = config.get("degradation", {})
    requested_families = analysis_cfg.get("eval_families", degradation_cfg.get("eval_families"))
    if requested_families:
        families = [str(item) for item in requested_families]
    elif degradation_cfg.get("enabled", False):
        families = [str(degradation_cfg.get("default_eval_family", "random_missing_months"))]
    else:
        families = ["clean"]
    configured_severities = analysis_cfg.get("eval_severities")
    severity_map: dict[str, list[str]] = {}
    for family in families:
        if isinstance(configured_severities, dict) and family in configured_severities:
            severity_map[family] = [str(item) for item in configured_severities[family]]
        elif isinstance(configured_severities, list):
            severity_map[family] = [str(item) for item in configured_severities]
        elif family == "clean":
            severity_map[family] = ["L0"]
        else:
            severity_map[family] = [str(item) for item in degradation_cfg["families"][family]["severities"].keys()]
    return families, severity_map
