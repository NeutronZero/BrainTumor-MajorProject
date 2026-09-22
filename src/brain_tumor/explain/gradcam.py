"""Grad-CAM observer for CLS-001 (Workstream 2, EXPL-001).

Strict observer contract: hooks only, no weight/architecture change, no
retraining. Target layer = last nn.Conv2d under model.features when the
architecture has one (torchvision ConvNeXt-Tiny layout => features.7.2.block.0);
architectures without a Conv2d under `features` fall back to the last
nn.Conv2d anywhere in the model (multi-arch robustness). Resolved
programmatically and recorded. Eval mode, deterministic given identical
input (no RNG in this path).

Boundary: the heatmap visualizes regions contributing to the classification
output. It does NOT prove the model "looks at the tumor" and carries no
clinical-meaning claim without separate validation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn


def resolve_target_layer(model: nn.Module) -> tuple[str, nn.Module]:
    """Resolve the Grad-CAM target layer: last Conv2d under `features`.

    Priority: (1) last nn.Conv2d whose dotted name starts with `features`
    — the frozen ConvNeXt-Tiny path (resolves to features.7.2.block.0);
    (2) if the architecture has no Conv2d under `features`, fall back to
    the last nn.Conv2d anywhere in the model; (3) raise RuntimeError only
    when the model contains zero nn.Conv2d modules.
    """
    found = [(n, m) for n, m in model.named_modules()
             if n.startswith("features") and isinstance(m, nn.Conv2d)]
    if found:
        return found[-1]
    any_conv = [(n, m) for n, m in model.named_modules()
                if isinstance(m, nn.Conv2d)]
    if not any_conv:
        # Real exception (not assert): must survive `python -O` deployment.
        raise RuntimeError("model has no nn.Conv2d — cannot resolve Grad-CAM target layer")
    return any_conv[-1]


def gradcam_heatmap(model: nn.Module, target: tuple[str, nn.Module], x: torch.Tensor,
                    class_idx: int) -> tuple[np.ndarray, str]:
    """CAM for class_idx (pre-softmax logit). Returns (H,W float32 in [0,1], target name)."""
    target_name, target_mod = target
    acts: dict = {}
    grads: dict = {}

    def fwd_hook(_m, _i, o):
        acts["v"] = o.detach()

    def bwd_hook(_m, _gi, go):
        grads["v"] = go[0].detach()

    h1 = target_mod.register_forward_hook(fwd_hook)
    h2 = target_mod.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(x)
        logits[0, class_idx].backward()
    finally:
        h1.remove()
        h2.remove()
    w = grads["v"].mean(dim=(2, 3), keepdim=True)
    cam = (w * acts["v"]).sum(dim=1, keepdim=True).clamp(min=0)
    cam = cam / (cam.amax(dim=(2, 3), keepdim=True) + 1e-12)
    cam = cam[0, 0].cpu().numpy().astype(np.float32)
    return cam, target_name


def upsample_cam(cam: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    from PIL import Image
    return np.asarray(Image.fromarray((cam * 255).astype("uint8")).resize(
        size, Image.BILINEAR), dtype=np.float32) / 255.0


def jet(m: np.ndarray) -> np.ndarray:
    """Deterministic JET colormap, no matplotlib dependency. In [0,1], RGB."""
    m = np.clip(m, 0, 1)
    r = np.clip(1.5 - np.abs(4 * m - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * m - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * m - 1), 0, 1)
    return np.stack([r, g, b], axis=-1)


def overlay(gray: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend uint8 grayscale base with JET heatmap. Returns uint8 RGB."""
    base = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    return (((1 - alpha) * base + alpha * jet(cam)) * 255).clip(0, 255).astype("uint8")


def cam_mass_in_bbox(cam: np.ndarray, bbox: tuple[int, int, int, int] | None) -> float | None:
    """Fraction of normalized-CAM mass inside an xyxy original-pixel bbox.

    Pure coordinate convention (verified: all-inside -> 1.0, all-outside ->
    0.0). Returns None when bbox is None. Descriptive statistic only.
    """
    if bbox is None:
        return None
    x0, y0, x1, y1 = bbox
    return round(float(cam[y0:y1, x0:x1].sum() / max(cam.sum(), 1e-12)), 4)
