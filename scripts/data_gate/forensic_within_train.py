"""forensic_within_train.py — reconcile 35-vs-34 within-train duplicate groups.

Deterministic report (no deletions, no relabeling):
  - every within-classification-train SHA group: hash, paths, labels, multiplicity
  - triplet brisc2025_train_04176/04177/04178: their SHAs, group membership,
    whether they bridge cross-split groups or form their own grouping records
  - cross-split group membership overlap with within-train groups

Outputs: outputs/data_gate_0/forensic_within_train.json + .md
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gate.common import outputs_dir, sha256_file, write_json  # noqa: E402
from data_gate.layout import (  # noqa: E402
    CLS_IMG_EXTS,
    cls_split_dirs,
    find_release_root,
    list_files,
)

TRIPLET = ("brisc2025_train_04176", "brisc2025_train_04177", "brisc2025_train_04178")


def main() -> dict:
    root = Path(__file__).resolve().parents[2]
    out = outputs_dir()
    release = find_release_root(root)
    assert release is not None, "release_not_found"
    cls = cls_split_dirs(release)
    train = list_files(cls["train"], CLS_IMG_EXTS)
    test = list_files(cls["test"], CLS_IMG_EXTS)

    train_map: dict[str, list[str]] = defaultdict(list)
    for p in train:
        train_map[sha256_file(p)].append(str(p.relative_to(root)))
    test_hashes = {sha256_file(p) for p in test}

    groups = sorted(
        ((h, sorted(v)) for h, v in train_map.items() if len(v) > 1),
        key=lambda kv: kv[1][0],
    )
    group_records = []
    for h, paths in groups:
        labels = sorted({Path(p).parent.name for p in paths})
        stems = sorted({Path(p).stem for p in paths})
        group_records.append(
            {
                "sha256": h,
                "multiplicity": len(paths),
                "labels": labels,
                "stems": stems,
                "paths": paths,
                "also_in_test": h in test_hashes,
            }
        )

    trip_info = {}
    for key in ("04176", "04177", "04178"):
        hits = [p for p in train if key in p.stem]
        stem = f"brisc2025_train_{key}"
        if not hits:
            trip_info[stem] = {"present_in_train": False}
            continue
        h = sha256_file(hits[0])
        members = sorted(train_map[h])
        trip_info[stem] = {
            "present_in_train": True,
            "sha256": h,
            "group_multiplicity": len(members),
            "group_members": [Path(m).stem for m in members],
            "group_labels": sorted({Path(m).parent.name for m in members}),
            "also_in_test": h in test_hashes,
        }

    trip_hashes = {v["sha256"] for v in trip_info.values() if v.get("present_in_train")}
    bridging = [g for g in group_records if g["sha256"] in trip_hashes and g["also_in_test"]]

    result = {
        "check": "forensic_within_train",
        "n_train_files": len(train),
        "n_unique_train_hashes": len(train_map),
        "n_within_train_groups": len(groups),
        "n_excess": sum(len(v) - 1 for _, v in groups),
        "groups": group_records,
        "triplet": trip_info,
        "triplet_distinct_hashes": len(trip_hashes),
        "triplet_bridges_cross_split": bool(bridging),
        "bridging_groups": bridging,
    }
    write_json("forensic_within_train.json", result)
    lines = [
        "# Forensic: within-train duplicate groups",
        f"groups={len(groups)} excess={result['n_excess']} unique={len(train_map)}",
        f"triplet distinct hashes={len(trip_hashes)} bridges_cross={bool(bridging)}",
        "",
    ]
    for g in group_records:
        lines.append(
            f"- {g['sha256'][:12]} x{g['multiplicity']} labels={g['labels']} "
            f"cross={g['also_in_test']} stems={g['stems']}"
        )
    (out / "forensic_within_train.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        f"forensic: groups={len(groups)} triplet_hashes={len(trip_hashes)} bridges={bool(bridging)}"
    )
    print(json.dumps(trip_info, indent=1))
    return result


if __name__ == "__main__":
    main()
