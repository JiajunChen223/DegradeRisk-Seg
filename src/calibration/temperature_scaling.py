from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass
class TemperatureScaler:
    temperature: float = 1.0

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, max_iter: int = 50) -> "TemperatureScaler":
        log_temp = torch.nn.Parameter(torch.zeros(1, dtype=logits.dtype, device=logits.device))
        optimizer = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=max_iter)
        criterion = torch.nn.BCEWithLogitsLoss()
        flat_logits = logits.reshape(-1)
        flat_targets = targets.reshape(-1).float()

        def closure() -> torch.Tensor:
            optimizer.zero_grad()
            temp = torch.exp(log_temp)
            loss = criterion(flat_logits / temp, flat_targets)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.temperature = float(torch.exp(log_temp).detach().cpu().item())
        return self

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / max(self.temperature, 1e-6)

    def state_dict(self) -> dict[str, float]:
        return {"temperature": float(self.temperature)}

    @classmethod
    def from_state_dict(cls, payload: dict[str, float]) -> "TemperatureScaler":
        return cls(temperature=float(payload["temperature"]))


class GroupedTemperatureScaler:
    def __init__(self, default_temperature: float = 1.0) -> None:
        self.default_temperature = default_temperature
        self.group_temperatures: dict[str, float] = {}

    def fit(self, logits: torch.Tensor, targets: torch.Tensor, groups: Iterable[str]) -> "GroupedTemperatureScaler":
        groups = list(groups)
        for group in sorted(set(groups)):
            mask = torch.tensor([item == group for item in groups], device=logits.device)
            if mask.sum() == 0:
                continue
            scaler = TemperatureScaler(temperature=self.default_temperature)
            scaler.fit(logits[mask], targets[mask])
            self.group_temperatures[group] = scaler.temperature
        return self

    def transform(self, logits: torch.Tensor, groups: Iterable[str]) -> torch.Tensor:
        scaled = []
        for logit, group in zip(logits, groups):
            temperature = self.group_temperatures.get(group, self.default_temperature)
            scaled.append(logit / max(temperature, 1e-6))
        return torch.stack(scaled, dim=0)

    def state_dict(self) -> dict[str, object]:
        return {
            "default_temperature": float(self.default_temperature),
            "group_temperatures": dict(self.group_temperatures),
        }

    @classmethod
    def from_state_dict(cls, payload: dict[str, object]) -> "GroupedTemperatureScaler":
        inst = cls(default_temperature=float(payload.get("default_temperature", 1.0)))
        inst.group_temperatures = {str(k): float(v) for k, v in dict(payload.get("group_temperatures", {})).items()}
        return inst


def apply_calibration(logits: torch.Tensor, calibrator: TemperatureScaler | GroupedTemperatureScaler | None, groups: list[str] | None = None) -> torch.Tensor:
    if calibrator is None:
        return logits
    if isinstance(calibrator, GroupedTemperatureScaler):
        if groups is None:
            raise ValueError("GroupedTemperatureScaler requires groups")
        return calibrator.transform(logits, groups)
    return calibrator.transform(logits)

