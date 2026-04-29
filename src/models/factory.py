from __future__ import annotations

from torch import nn

from src.models.backbones.resunet import ResUNet


def build_model(model_cfg: dict, in_channels: int) -> nn.Module:
    name = model_cfg["name"].lower()
    base_channels = int(model_cfg.get("base_channels", 32))
    depth = int(model_cfg.get("depth", 4))
    dropout = float(model_cfg.get("dropout", 0.1))
    norm = str(model_cfg.get("norm", "batch"))
    norm_groups = int(model_cfg.get("norm_groups", 8))
    if name == "resunet":
        return ResUNet(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            dropout=dropout,
            norm=norm,
            norm_groups=norm_groups,
        )
    raise KeyError(f"Unsupported paper-only model: {model_cfg['name']}. Expected 'resunet'.")
