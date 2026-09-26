"""inspect_masks.py — Revision A: evidence-preserving mask audit (§7).

Two-stage, never collapsing silently:
  STAGE 1 — characterize actual bytes per mask: mode, size, dtype, unique
    values/frequencies, min/max, intermediate-value fraction
    (pixels not in {0,255} / total), paired-image dimension match.
  STAGE 2 — evaluate whether `value > 127 -> foreground` is a valid binary
    interpretation of THIS release.

Verdicts:
  binary            — strict 0/255 (or 0/1), no intermediates.
  binary_intent_aa  — single-channel, range within [0,255], global intermediate
                      fraction <= cap (gate_thresholds.yaml), pairing + dims OK.
                      Convention `mask > 127 -> 1 else 0` recorded, NOT applied
                      to rewrite any file.
  multiclass        — small discrete corpus value set with substantial
                      intermediates (needs explicit design; no silent collapse).
  unknown           — anything else.

PASS only for binary / binary_intent_aa with intact pairing and zero
dimension mismatches. FAIL otherwise — including absent masks, in which case
§12 re-scopes segmentation to a stretch goal (reported, not asserted).
Also reports mask-SHA duplicate stats and image->divergent-mask counts.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gate.common import sha256_file, write_json  # noqa: E402
from data_gate.layout import find_release_root, seg_split_dirs  # noqa: E402


def _analyze_mask(p: Path):
    """Return dict of Stage-1 facts, or None if unreadable."""
    from PIL import Image

    with Image.open(p) as im:
        mode, size = im.mode, im.size
        g = im.convert("L") if mode != "L" else im
        hist = g.histogram()  # 256 bins
        total = sum(hist)
        c0, c255 = hist[0], hist[255]
        inter = total - c0 - c255
        fg127 = sum(hist[128:])
        uniques = [i for i, c in enumerate(hist) if c]
        return {
            "mode": mode,
            "size": [size[0], size[1]],
            "total": total,
            "c0": c0,
            "c255": c255,
            "intermediate": inter,
            "fg127": fg127,
            "n_unique": len(uniques),
            "min": uniques[0],
            "max": uniques[-1],
            "uniques_sample": uniques[:32],
        }


def _img_size(p: Path):
    try:
        from PIL import Image

        with Image.open(p) as im:
            return [im.size[0], im.size[1]]
    except Exception:
        return None


def main() -> dict:
    root = Path(__file__).resolve().parents[2]
    release = find_release_root(root)
    thr_cfg = yaml.safe_load(open(root / "configs" / "data" / "gate_thresholds.yaml")) or {}
    inter_cap = float(thr_cfg.get("mask_intermediate_fraction_cap", 0.02))
    allowed = set(thr_cfg.get("mask_allowed_encodings", ["binary", "binary_intent_aa"]))
    if release is None:
        result = {
            "check": "masks",
            "pass": False,
            "mask_encoding": "unknown",
            "n_masks": 0,
            "reason": "release_not_found",
        }
        write_json("mask_report.json", result)
        print("masks: FAIL (release_not_found)")
        return result

    try:
        import PIL  # noqa: F401
    except ImportError:
        result = {
            "check": "masks",
            "pass": False,
            "mask_encoding": "unknown",
            "reason": "PIL_unavailable_cannot_verify",
        }
        write_json("mask_report.json", result)
        print("masks: FAIL (PIL unavailable — fail-closed)")
        return result

    seg = seg_split_dirs(release)
    modes: Counter = Counter()
    n_unique_hist: Counter = Counter()
    tot_px = tot_inter = 0
    corpus_values: Counter = Counter()
    n_checked = n_fg = n_empty = dim_mismatch = unreadable = 0
    missing_mask: list[str] = []
    missing_image: list[str] = []
    mask_hashes: dict[str, str] = {}
    img_to_masks: dict[str, set[str]] = {}

    for split in ("train", "test"):
        images = {p.stem: p for p in sorted(seg[split]["images"].glob("*.jpg"))}
        masks = {p.stem: p for p in sorted(seg[split]["masks"].glob("*.png"))}
        for stem in sorted(set(images) | set(masks)):
            if stem not in masks:
                missing_mask.append(f"{split}/{stem}")
                continue
            if stem not in images:
                missing_image.append(f"{split}/{stem}")
                continue
            mp = masks[stem]
            try:
                a = _analyze_mask(mp)
            except Exception:
                unreadable += 1
                continue
            n_checked += 1
            modes[a["mode"]] += 1
            n_unique_hist[a["n_unique"]] += 1
            tot_px += a["total"]
            tot_inter += a["intermediate"]
            for v in a["uniques_sample"]:
                corpus_values[v] += 1
            # foreground under the evaluated convention (>127):
            if a["fg127"] > 0:
                n_fg += 1
            else:
                n_empty += 1
            isize = _img_size(images[stem])
            if isize != a["size"]:
                dim_mismatch += 1
            try:
                mh = sha256_file(mp)
                ih = sha256_file(images[stem])
                mask_hashes[mh] = mask_hashes.get(mh, mh)
                img_to_masks.setdefault(ih, set()).add(mh)
            except Exception:
                pass

    n_masks_total = sum(1 for s in ("train", "test") for _ in seg[s]["masks"].glob("*.png"))
    glob_inter_frac = (tot_inter / tot_px) if tot_px else 1.0
    divergent = sum(1 for v in img_to_masks.values() if len(v) > 1)

    # Stage 2 evaluation
    single_channel = set(modes) <= {"L", "P", "1"}
    small_discrete = len(corpus_values) <= 8
    if n_checked == 0:
        encoding, ok, reason = "unknown", False, "masks_absent_or_not_found"
    elif not single_channel:
        encoding, ok, reason = "unknown", False, "non_single_channel_modes"
    elif glob_inter_frac == 0.0 and max(n_unique_hist) <= 2:
        encoding, ok, reason = "binary", True, None
    elif glob_inter_frac <= inter_cap:
        encoding, ok, reason = "binary_intent_aa", True, None
    elif small_discrete and glob_inter_frac > inter_cap:
        encoding, ok, reason = "multiclass", False, "multiclass_needs_explicit_design"
    else:
        encoding, ok, reason = "unknown", False, "mask_encoding_unknown"

    pairing_ok = not missing_mask and not missing_image and unreadable == 0
    if ok and (not pairing_ok or dim_mismatch):
        ok = False
        reason = "broken_image_mask_pairing" if not pairing_ok else "dimension_mismatch"
    if encoding not in allowed:
        ok = False
        if reason is None:
            reason = "encoding_not_in_allowlist"

    result = {
        "check": "masks",
        "pass": ok,
        "mask_encoding": encoding,
        "stage1": {
            "n_masks_on_disk": n_masks_total,
            "n_checked": n_checked,
            "modes": dict(modes),
            "n_unique_histogram": {str(k): v for k, v in sorted(n_unique_hist.items())},
            "global_intermediate_fraction": glob_inter_frac,
            "intermediate_cap": inter_cap,
            "corpus_distinct_values_seen": len(corpus_values),
        },
        "convention_evaluated": "mask > 127 -> 1 else 0 (documented, files untouched)",
        "coverage_after_convention": {"with_foreground": n_fg, "empty": n_empty},
        "pairing": {
            "missing_mask": missing_mask[:10],
            "n_missing_mask": len(missing_mask),
            "missing_image": missing_image[:10],
            "n_missing_image": len(missing_image),
            "dimension_mismatches": dim_mismatch,
            "unreadable": unreadable,
        },
        "mask_sha": {
            "n_unique_mask_hashes": len(mask_hashes),
            "n_images_with_divergent_masks": divergent,
        },
        "reason": reason,
        "gate_effect": ("FAIL -> segmentation becomes stretch goal (§12)" if not ok else None),
        "note": "Binary-intent with antialiased edges is a release property, "
        "not a binarization decision by this gate.",
    }
    write_json("mask_report.json", result)
    print(
        f"masks: {'PASS' if ok else 'FAIL'} encoding={encoding} "
        f"inter_frac={glob_inter_frac:.4f} fg={n_fg} empty={n_empty}"
    )
    return result


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["pass"] else 2)
