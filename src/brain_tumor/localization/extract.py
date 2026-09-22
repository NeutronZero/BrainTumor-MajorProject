"""Localization extraction — frozen semantics (§48-49 + contract freeze).

threshold 0.5, component_policy largest_plus_total, min_area_pixels 10.
bbox xyxy ints + centroid floats in ORIGINAL input pixels; empty -> None/None/0.

Note: the contract reserves a warning code "oversized_component_filtered"
(contracts.py WarningCode). It is intentionally NEVER emitted by this frozen
extractor — no size-based component filtering is implemented here. Adding
filtering would change frozen localization outputs (which components qualify
for bbox/centroid/area), so the code remains reserved/dead vocabulary.
"""

from __future__ import annotations

import numpy as np

from brain_tumor.contracts import LocalizationResult


def extract(prob_map: np.ndarray, orig_hw: tuple[int, int],
            threshold: float = 0.5, min_area: int = 10) -> tuple[LocalizationResult, list[str]]:
    from scipy import ndimage
    H, W = orig_hw
    h, w = prob_map.shape[-2:]
    binary = (prob_map > threshold).astype(np.uint8)
    lab, n = ndimage.label(binary)
    if n == 0:
        return LocalizationResult(bbox=None, centroid=None, area_pixels=0), []
    areas = ndimage.sum(binary, lab, range(1, n + 1))
    keep = [(i + 1, a) for i, a in enumerate(areas) if a >= min_area]
    warnings: list[str] = []
    if not keep:
        return LocalizationResult(bbox=None, centroid=None, area_pixels=0), []
    if len(keep) > 1:
        warnings.append("multiple_components")
    total = int(sum(a for _, a in keep))
    lid = max(keep, key=lambda t: t[1])[0]
    ys, xs = np.where(lab == lid)
    cy, cx = float(ys.mean()), float(xs.mean())
    # map model-frame coords back to original pixels
    sy, sx = H / h, W / w
    x0, y0, x1, y1 = int(xs.min() * sx), int(ys.min() * sy), int((xs.max() + 1) * sx), int((ys.max() + 1) * sy)
    loc = LocalizationResult(bbox=(x0, y0, x1, y1),
                             centroid=(float(cx * sx), float(cy * sy)),
                             area_pixels=total)
    return loc, warnings
