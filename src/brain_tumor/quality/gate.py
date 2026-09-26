"""Deterministic input-quality gate (Workstream 1, SB-1 fallback intact).

Pure engineering checks on the input bytes/pixels. No model, no learning,
no baseline contact. Fixed engineering constants with rationale inline;
thresholds are NOT tuned on the locked test set (or on validation — val is
used only to REPORT the false-rejection rate).

Each check returns a coded reason on failure. Verdict = pass iff all pass.
Codes are stable API: Q01..Q09. Q10 added for MIME sniffing defense-in-depth.
"""

from __future__ import annotations

import io

import filetype
import numpy as np

# Fixed constants (engineering rationale in comments).
MIN_SIDE_PX = 64  # far below smallest released image (174); catches icons/thumbnails
MAX_ASPECT = 8.0  # catches stripe/panoramic anomalies; false-reject
# rate measured on clean val (must stay ~0)
MIN_STD = 1.0  # 0-255 scale; blank/uniform images sit at ~0
MIN_RANGE = 8.0  # max-min intensity floor
MIN_MEAN, MAX_MEAN = 2.0, 253.0  # all-black / all-white guards
MIN_FOREGROUND_FRAC = 0.02  # pixels > 10/255; head MRI always exceeds this
FOREGROUND_LEVEL = 10.0
MAX_BYTES = 10 * 1024 * 1024  # matches API limit
_ALLOWED_MIME = {"image/jpeg", "image/png", "image/bmp", "image/tiff"}


def _reasons() -> list[str]:
    return []


def assess(data: bytes) -> dict:
    """Assess raw upload bytes. Returns {verdict, failed: [codes], facts}."""
    failed = _reasons()
    facts: dict = {"n_bytes": len(data)}
    if len(data) > MAX_BYTES:
        failed.append("Q08_size_limit")
        return {"verdict": "reject", "failed": failed, "facts": facts}
    # Q10: MIME sniffing via filetype (defense-in-depth against content-type spoofing)
    kind = filetype.guess(data[:261])
    if kind is None or kind.mime not in _ALLOWED_MIME:
        failed.append("Q10_mime_mismatch")
        # Don't return early; continue to collect other quality facts
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:  # noqa: BLE001
        failed.append("Q01_undecodable")
        return {"verdict": "reject", "failed": failed, "facts": facts}
    w, h = img.size
    facts.update({"mode": img.mode, "width": w, "height": h})
    if min(w, h) < MIN_SIDE_PX:
        failed.append("Q03_too_small")
    if max(w, h) / max(min(w, h), 1) > MAX_ASPECT:
        failed.append("Q09_extreme_aspect")
    try:
        a = np.asarray(img.convert("L"), dtype=np.float64)
    except Exception:  # noqa: BLE001
        failed.append("Q02_unreadable_pixels")
        return {"verdict": "reject", "failed": failed, "facts": facts}
    if not np.all(np.isfinite(a)):
        failed.append("Q04_nonfinite")
        return {"verdict": "reject", "failed": failed, "facts": facts}
    facts.update(
        {
            "mean": round(float(a.mean()), 3),
            "std": round(float(a.std()), 3),
            "min": float(a.min()),
            "max": float(a.max()),
        }
    )
    if float(a.std()) < MIN_STD or float(a.max() - a.min()) < MIN_RANGE:
        failed.append("Q05_blank_or_uniform")
    if not (MIN_MEAN <= float(a.mean()) <= MAX_MEAN):
        failed.append("Q06_intensity_out_of_range")
    fg = float((a > FOREGROUND_LEVEL).mean())
    facts["foreground_frac"] = round(fg, 4)
    if fg < MIN_FOREGROUND_FRAC:
        failed.append("Q07_insufficient_content")
    return {"verdict": "accept" if not failed else "reject", "failed": failed, "facts": facts}
