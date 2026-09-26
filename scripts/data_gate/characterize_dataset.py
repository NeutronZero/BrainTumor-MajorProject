"""characterize_dataset.py — Revision A: scoped to classification_task/{train,test}.

Only class-folder .jpg images count toward the classification population.
Segmentation trees and manifests are audited by their own checks, never here.
FAIL (fail-closed) on: release absent, empty population, unexpected labels,
or train/test counts differing from configs/data/brisc2025.yaml expected.
Raw folder names are preserved alongside normalized vocabulary mapping.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gate.common import write_json  # noqa: E402
from data_gate.layout import (  # noqa: E402
    CLASS_ALIAS,
    CLS_IMG_EXTS,
    KNOWN_RAW_LABELS,
    cls_split_dirs,
    find_release_root,
    list_files,
)


def _img_size(p: Path):
    try:
        from PIL import Image

        with Image.open(p) as im:
            return im.size, im.format
    except Exception:
        return None, None


def main() -> dict:
    root = Path(__file__).resolve().parents[2]
    release = find_release_root(root)
    if release is None:
        result = {
            "check": "characterize",
            "pass": False,
            "scope": "classification_task",
            "reason": "release_not_found",
            "note": "Expected <data>/brisc2025/classification_task/{train,test}.",
        }
        write_json("dataset_summary.json", result)
        print("characterize: FAIL (release_not_found)")
        return result

    expected = (yaml.safe_load(open(root / "configs" / "data" / "brisc2025.yaml")) or {}).get(
        "expected", {}
    )
    exp_train = int(expected.get("train_pool", 5000))
    exp_test = int(expected.get("official_test", 1000))

    per_split: dict[str, dict] = {}
    unexpected: Counter = Counter()
    fmts: Counter = Counter()
    sizes: list[list[int]] = []
    total = 0
    for split, d in cls_split_dirs(release).items():
        files = list_files(d, CLS_IMG_EXTS)
        counts_raw: Counter = Counter()
        counts_norm: Counter = Counter()
        for p in files:
            raw = p.parent.name
            counts_raw[raw] += 1
            if raw in KNOWN_RAW_LABELS:
                counts_norm[CLASS_ALIAS[raw]] += 1
            else:
                unexpected[f"{split}/{raw}"] += 1
            wh, fmt = _img_size(p)
            if wh:
                sizes.append([wh[0], wh[1]])
            if fmt:
                fmts[fmt] += 1
            fmts[p.suffix.lower()] += 1
        per_split[split] = {
            "n": len(files),
            "raw_counts": dict(counts_raw),
            "class_counts": dict(counts_norm),
        }
        total += len(files)

    count_ok = per_split["train"]["n"] == exp_train and per_split["test"]["n"] == exp_test
    ok = total > 0 and not unexpected and count_ok
    if not total:
        reason = "dataset_not_present"
    elif unexpected:
        reason = "unexpected_labels"
    elif not count_ok:
        reason = "count_mismatch_vs_expected"
    else:
        reason = None
    result = {
        "check": "characterize",
        "pass": ok,
        "scope": "classification_task/{train,test}",
        "release": str(release.relative_to(root)),
        "n_images": total,
        "per_split": per_split,
        "unexpected_labels": dict(unexpected),
        "expected": {"train": exp_train, "test": exp_test},
        "count_match": count_ok,
        "formats": dict(fmts),
        "resolution": {
            "n_measured": len(sizes),
            "min": [int(min(s[0] for s in sizes)), int(min(s[1] for s in sizes))]
            if sizes
            else None,
            "max": [int(max(s[0] for s in sizes)), int(max(s[1] for s in sizes))]
            if sizes
            else None,
        },
        "reason": reason,
        "vocabulary_note": "Raw folder 'no_tumor' maps to contract 'notumor'; raw names preserved above.",
    }
    write_json("dataset_summary.json", result)
    print(
        f"characterize: {'PASS' if ok else 'FAIL'} "
        f"train={per_split['train']['n']} test={per_split['test']['n']}"
    )
    return result


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["pass"] else 2)
