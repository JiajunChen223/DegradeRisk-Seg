from __future__ import annotations

import json

import torch
import _bootstrap  # noqa: F401

from scripts.common import (
    build_model_and_device,
    checkpoint_artifact_path,
    experiment_run_name,
    load_experiment,
    make_loader,
    maybe_prepare_data,
    parse_args,
    prepare_model_inputs,
    resolve_pos_weight_value,
    save_checkpoint,
    summary_provenance,
)
from src.evaluators.calibration import compute_binary_calibration_metrics
from src.evaluators.segmentation import batch_dice_iou
from src.utils.experiment import save_metrics_summary


def evaluate(model: torch.nn.Module, loader, device: torch.device, config: dict) -> dict[str, float]:
    model.eval()
    logits_all = []
    targets_all = []
    dice_total = 0.0
    iou_total = 0.0
    batches = 0
    with torch.no_grad():
        for batch in loader:
            x = prepare_model_inputs(batch["x"], device, config)
            y = batch["y"].to(device, non_blocking=True)
            logits = model(x)
            seg = batch_dice_iou(logits, y)
            dice_total += seg["dice"]
            iou_total += seg["iou"]
            logits_all.append(logits.detach().cpu())
            targets_all.append(y.detach().cpu().unsqueeze(1))
            batches += 1
    logits = torch.cat(logits_all, dim=0)
    targets = torch.cat(targets_all, dim=0)
    cal = compute_binary_calibration_metrics(logits, targets)
    return {
        "dice": dice_total / max(batches, 1),
        "iou": iou_total / max(batches, 1),
        **cal,
    }


def main() -> None:
    args = parse_args("Train Plot-Rice baseline")
    config, run_dir = load_experiment(args)
    maybe_prepare_data(config)
    model, device = build_model_and_device(config, args.device)
    train_cfg = config["train"]
    train_loader = make_loader(config, config["split"]["train"], shuffle=True)
    val_loader = make_loader(config, config["split"]["val"])
    max_epochs = int(train_cfg.get("max_epochs", train_cfg.get("epochs", 1)))
    min_epochs = int(train_cfg.get("min_epochs", 1))
    early_stop_patience = int(train_cfg.get("early_stop_patience", max_epochs))
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    effective_batch_size = int(train_cfg.get("batch_size", 1)) * grad_accum_steps
    pos_weight_value = resolve_pos_weight_value(config, train_loader)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler_cfg = train_cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 3)),
        min_lr=float(scheduler_cfg.get("min_lr", 1.0e-5)),
    )
    best_dice = -1.0
    best_metrics = {}
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_trained = 0
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0
        batches = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader, start=1):
            x = prepare_model_inputs(batch["x"], device, config)
            y = batch["y"].to(device, non_blocking=True).unsqueeze(1)
            logits = model(x)
            raw_loss = criterion(logits, y)
            loss = raw_loss / grad_accum_steps
            loss.backward()
            if batch_idx % grad_accum_steps == 0 or batch_idx == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(raw_loss.item())
            batches += 1
        metrics = evaluate(model, val_loader, device, config)
        metrics["train_loss"] = running_loss / max(batches, 1)
        metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        scheduler.step(metrics["dice"])
        epochs_trained = epoch + 1
        if metrics["dice"] > best_dice + 1.0e-8:
            best_dice = metrics["dice"]
            best_metrics = metrics
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            save_checkpoint(model, config, run_dir / "best_dice.pt")
            save_checkpoint(model, config, checkpoint_artifact_path(config, run_name=experiment_run_name(config), kind="best-dice"))
        else:
            epochs_without_improvement += 1
        if epochs_trained >= min_epochs and epochs_without_improvement >= early_stop_patience:
            break
    save_checkpoint(model, config, run_dir / "last.pt")
    checkpoint_source = checkpoint_artifact_path(config, run_name=experiment_run_name(config), kind="best-dice")
    summary = {
        **summary_provenance(config, checkpoint_source=checkpoint_source),
        "model": config["model"]["name"],
        "train_mode": config["train"]["name"],
        "max_epochs": max_epochs,
        "min_epochs": min_epochs,
        "epochs_trained": epochs_trained,
        "best_epoch": best_epoch,
        "early_stop_patience": early_stop_patience,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": effective_batch_size,
        "scheduler": str(scheduler_cfg.get("name", "reduce_on_plateau")),
        "pos_weight": pos_weight_value,
        "clean_dice": best_metrics.get("dice", 0.0),
        "clean_iou": best_metrics.get("iou", 0.0),
        "ece": best_metrics.get("ece", 0.0),
        "nll": best_metrics.get("nll", 0.0),
        "brier": best_metrics.get("brier", 0.0),
    }
    save_metrics_summary(summary, run_dir)
    (run_dir / "train.log").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
