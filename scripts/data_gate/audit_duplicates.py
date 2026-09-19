"""audit_duplicates.py — Revision A: task-aware exact-SHA256 audit (§9).

Categories (always all reported):
  1. within classification train
  2. within classification test
  3. across classification train <-> test            <- CRITICAL (hard finding)
  4. within segmentation train (images)
  5. within segmentation test (images)
  6. across segmentation train <-> test (images)     <- CRITICAL (hard finding)
  7. classification <-> segmentation correspondence  <- REPORTED, not contamination
     (segmentation is a documented subset/task representation of the same
     dataset; same-byte presence across the two trees is expected mechanics)

FAIL (fail-closed) if any CRITICAL category exceeds
configs/data/gate_thresholds.yaml:duplicate_hash_threshold (default 0),
or if the release is absent. Within-partition groups (1,2,4,5) are findings:
reported with counts and samples, never auto-deleted, never silently dropped.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gate.common import sha256_file, write_json  # noqa: E402
from data_gate.layout import (  # noqa: E402
    CLS_IMG_EXTS,
    cls_split_dirs,
    find_release_root,
    list_files,
    seg_split_dirs,
)


def _sha_map(files: list[Path]) -> dict[str, list[str]]:
    root = Path(__file__).resolve().parents[2]
    buckets: dict[str, list[str]] = defaultdict(list)
    for p in files:
        try:
            buckets[sha256_file(p)].append(str(p.relative_to(root)))
        except Exception:
            pass
    return buckets


def _within(groups_map: dict[str, list[str]]) -> dict:
    dups = {h: v for h, v in groups_map.items() if len(v) > 1}
    return {"n_files": sum(len(v) for v in groups_map.values()),
            "n_unique": len(groups_map),
            "n_groups": len(dups),
            "n_excess": sum(len(v) - 1 for v in dups.values()),
            "groups_sample": list(dups.values())[:10]}


def main() -> dict:
    root = Path(__file__).resolve().parents[2]
    release = find_release_root(root)
    thr = int((yaml.safe_load(open(root / "configs" / "data" / "gate_thresholds.yaml")) or {}).get(
        "duplicate_hash_threshold", 0))
    if release is None:
        result = {"check": "exact_duplicates", "pass": False, "reason": "release_not_found"}
        write_json("duplicate_report.json", result)
        print("duplicates: FAIL (release_not_found)")
        return result

    cls = cls_split_dirs(release)
    seg = seg_split_dirs(release)
    cls_train = list_files(cls["train"], CLS_IMG_EXTS)
    cls_test = list_files(cls["test"], CLS_IMG_EXTS)
    seg_train = list_files(seg["train"]["images"], CLS_IMG_EXTS)
    seg_test = list_files(seg["test"]["images"], CLS_IMG_EXTS)

    m_ct, m_ce = _sha_map(cls_train), _sha_map(cls_test)
    m_st, m_se = _sha_map(seg_train), _sha_map(seg_test)

    cat1, cat2, cat4, cat5 = _within(m_ct), _within(m_ce), _within(m_st), _within(m_se)

    cross_cls = sorted(set(m_ct) & set(m_ce))
    cross_seg = sorted(set(m_st) & set(m_se))
    cat3 = {"n_shared_hashes": len(cross_cls),
            "n_train_files": sum(len(m_ct[h]) for h in cross_cls),
            "n_test_files": sum(len(m_ce[h]) for h in cross_cls),
            "hashes_sample": cross_cls[:10]}
    cat6 = {"n_shared_hashes": len(cross_seg),
            "n_train_files": sum(len(m_st[h]) for h in cross_seg),
            "n_test_files": sum(len(m_se[h]) for h in cross_seg),
            "hashes_sample": cross_seg[:10]}

    # Category 7: subset correspondence (reported only)
    cls_all = set(m_ct) | set(m_ce)
    seg_all = set(m_st) | set(m_se)
    matched = seg_all & cls_all
    cat7 = {"n_seg_unique": len(seg_all), "n_cls_unique": len(cls_all),
            "n_seg_hashes_found_in_classification": len(matched),
            "n_seg_hashes_only_in_segmentation": len(seg_all - cls_all),
            "note": "Same-byte classification<->segmentation presence is expected "
                    "subset mechanics, NOT contamination."}

    critical = cat3["n_shared_hashes"] + cat6["n_shared_hashes"]
    ok = (len(cls_train) + len(cls_test) > 0) and critical <= thr
    reason = None if ok else ("dataset_not_present" if not (cls_train or cls_test)
                              else "cross_split_contamination")
    result = {"check": "exact_duplicates", "pass": ok,
              "threshold_critical_cross": thr, "critical_cross_total": critical,
              "cat1_within_cls_train": cat1, "cat2_within_cls_test": cat2,
              "cat3_cross_cls_train_test_CRITICAL": cat3,
              "cat4_within_seg_train": cat4, "cat5_within_seg_test": cat5,
              "cat6_cross_seg_train_test_CRITICAL": cat6,
              "cat7_cls_seg_correspondence_report_only": cat7,
              "reason": reason}
    write_json("duplicate_report.json", result)
    print(f"duplicates: {'PASS' if ok else 'FAIL'} "
          f"cross_cls={cat3['n_shared_hashes']} cross_seg={cat6['n_shared_hashes']}")
    return result


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["pass"] else 2)
