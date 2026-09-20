"""Attention U-Net — SEG-002 candidate (GEN-001).

Single controlled delta vs SEG-001 (`unet.py`, untouched): additive attention
gates (Oktay et al. 2018) on the four skip connections. Encoder blocks,
upsampling, head, tensor I/O contract, and init convention are identical.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from brain_tumor.segmentation.unet import DoubleConv


class AttentionGate(nn.Module):
    """Additive gate: alpha = sigmoid(psi(ReLU(theta_x(x) + phi_g(g))))."""

    def __init__(self, c_x: int, c_g: int, f_int: int | None = None):
        super().__init__()
        f = f_int or max(1, c_x // 2)
        self.theta_x = nn.Conv2d(c_x, f, kernel_size=2, stride=2, bias=True)
        self.phi_g = nn.Conv2d(c_g, f, kernel_size=1, stride=2, bias=True)
        self.psi = nn.Conv2d(f, 1, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        tx = self.theta_x(x)
        pg = self.phi_g(g)
        if pg.shape[-2:] != tx.shape[-2:]:  # odd-size guard (cf. UNet forward)
            pg = F.interpolate(pg, size=tx.shape[-2:], mode="bilinear",
                               align_corners=False)
        a = self.relu(tx + pg)
        a = torch.sigmoid(self.psi(a))
        a = F.interpolate(a, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * a


class AttentionUNet(nn.Module):
    """Same channel topology as SEG-001 UNet; skips pass through gates."""

    def __init__(self, in_ch: int = 1, base: tuple[int, ...] = (64, 128, 256, 512)):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()
        c = in_ch
        for b in base:
            self.downs.append(DoubleConv(c, b))
            self.pools.append(nn.MaxPool2d(2))
            c = b
        self.bottleneck = DoubleConv(c, c * 2)  # 1024
        rev = list(reversed(base))  # [512, 256, 128, 64]
        self.upconvs = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.convs = nn.ModuleList()
        c = c * 2  # 1024
        for b in rev:
            self.upconvs.append(nn.ConvTranspose2d(c, b, 2, stride=2))
            self.gates.append(AttentionGate(c_x=b, c_g=b))
            self.convs.append(DoubleConv(c, b))  # concat(gated_skip, up) has c ch
            c = b
        self.final = nn.Conv2d(c, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for d, p in zip(self.downs, self.pools):
            x = d(x)
            skips.append(x)
            x = p(x)
        x = self.bottleneck(x)
        for up, gate, conv, skip in zip(self.upconvs, self.gates, self.convs,
                                        reversed(skips)):
            g = up(x)
            if g.shape[-2:] != skip.shape[-2:]:
                g = F.interpolate(g, size=skip.shape[-2:], mode="bilinear",
                                  align_corners=False)
            x = conv(torch.cat([gate(skip, g), g], dim=1))
        return self.final(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
