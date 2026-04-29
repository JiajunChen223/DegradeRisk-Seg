from __future__ import annotations

import torch
from torch import nn


def _make_norm(num_channels: int, norm: str = "group", norm_groups: int = 8) -> nn.Module:
    norm_name = norm.lower()
    if norm_name == "group":
        groups = min(norm_groups, num_channels)
        while groups > 1 and (num_channels % groups != 0):
            groups -= 1
        return nn.GroupNorm(num_groups=max(groups, 1), num_channels=num_channels)
    if norm_name == "batch":
        return nn.BatchNorm2d(num_channels)
    raise KeyError(f"Unsupported norm type: {norm}")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm: str = "group", norm_groups: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = _make_norm(out_channels, norm=norm, norm_groups=norm_groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = _make_norm(out_channels, norm=norm, norm_groups=norm_groups)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = nn.functional.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return nn.functional.relu(x + residual, inplace=True)


class ResUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
        norm: str = "group",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        channels = [base_channels * (2**idx) for idx in range(depth)]
        self.encoders = nn.ModuleList()
        prev = in_channels
        for ch in channels:
            self.encoders.append(ResidualBlock(prev, ch, norm=norm, norm_groups=norm_groups))
            prev = ch
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ResidualBlock(channels[-1], channels[-1] * 2, norm=norm, norm_groups=norm_groups)
        self.dropout = nn.Dropout2d(dropout)
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        decoder_in = channels[-1] * 2
        for ch in reversed(channels):
            self.upconvs.append(nn.ConvTranspose2d(decoder_in, ch, kernel_size=2, stride=2))
            self.decoders.append(ResidualBlock(ch * 2, ch, norm=norm, norm_groups=norm_groups))
            decoder_in = ch
        self.head = nn.Conv2d(base_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.dropout(self.bottleneck(x))
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)
        return self.head(x)
