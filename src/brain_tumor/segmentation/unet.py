"""Vanilla 2D U-Net — SEG-001 baseline (fresh implementation)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: tuple[int, ...] = (64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = in_ch
        for b in base:
            self.downs.append(DoubleConv(c, b))
            self.pools.append(nn.MaxPool2d(2))
            c = b
        self.bottleneck = DoubleConv(c, c * 2)
        self.ups = nn.ModuleList()
        rev = list(reversed(base))
        c = c * 2
        for b in rev:
            self.ups.append(nn.ConvTranspose2d(c, b, 2, stride=2))
            self.ups.append(DoubleConv(c, b))
            c = b
        self.final = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        skips = []
        for d, p in zip(self.downs, self.pools, strict=True):
            x = d(x)
            skips.append(x)
            x = p(x)
        x = self.bottleneck(x)
        for i, b in enumerate(reversed(skips)):
            up = self.ups[2 * i]
            conv = self.ups[2 * i + 1]
            x = up(x)
            if x.shape[-2:] != b.shape[-2:]:
                import torch.nn.functional as F

                x = F.interpolate(x, size=b.shape[-2:], mode="bilinear", align_corners=False)
            x = conv(torch.cat([b, x], dim=1))
        return self.final(x)


def dice_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    dice_w: float = 1.0,
    bce_w: float = 1.0,
    smooth: float = 1e-6,
):
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + smooth) / (
        probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + smooth
    )
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none").mean(
        dim=(1, 2, 3)
    )
    return (dice_w * dice + bce_w * bce).mean()
