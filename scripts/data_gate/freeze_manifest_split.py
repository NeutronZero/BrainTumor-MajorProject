"""freeze_manifest_split.py — immutable Gate-0 provenance + 4000/1000 split.

Authorized by PI CONDITIONAL-PASS verdict. No model work. Rules:
  R0 pool = 5000 official train files; official test 1000 LOCKED, untouched.
  R1 units are byte-SHA groups (atomic: same bytes never split train/val).
  R2 contaminated units (7 cross-split hashes) are RETAINED in the pool with
     flag cross_split_contaminated=1 and pre-assigned to project-train, so the
     project-validation set contains no test-contaminated bytes (calibration
     and threshold selection must have no test feedback).
  R3 remaining units: stratify class x plane (class-only fallback for
     multi-plane units), seed-42 hash-ordered, per-stratum quotas via largest
     remainder summing to exactly 4000 train / 1000 val files.
  R4 everything frozen + SHA-hashed. Fail-closed: any inconsistency -> exit 2.

Outputs (outputs/data_gate_0/):
  cross_split_exclusion_list.json  (7 hashes + train/test paths; sensitivity N=993)
  project_manifest.csv             (6000 rows + flags + split assignment)
  project_split.json               (membership lists + hashes + quotas)
  project_split_report.md
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gate.common import outputs_dir, sha256_file, write_json  # noqa: E402
from data_gate.layout import (  # noqa: E402
    CLASS_ALIAS,
    CLS_IMG_EXTS,
    cls_split_dirs,
    find_release_root,
    list_files,
)

SEED = 42
N_TRAIN, N_VAL = 4000, 1000
TUMOR = {"gl": "glioma", "me": "meningioma", "pi": "pituitary", "no": "notumor"}
VIEWS = {"ax", "co", "sa"}


def parse(stem: str) -> tuple[str, str]:
    parts = stem.split("_")  # brisc2025_{split}_{idx}_{tumor}_{view}_t1
    return TUMOR[parts[3]], parts[4]


def order_key(seed: int, unit_sha: str) -> str:
    return hashlib.sha256(f"{seed}:{unit_sha}".encode()).hexdigest()


def subset_for_quota(sizes: list[int], q: int) -> set[int] | None:
    """Exact subset summing to q via bitset DP. Deterministic: backtracking
    skips later units whenever possible, so train favors earlier hash order.
    Returns chosen indices, or None if q unattainable."""
    total = sum(sizes)
    if q < 0 or q > total:
        return None
    mask_all = (1 << (total + 1)) - 1
    snaps = [1]
    for sz in sizes:
        snaps.append(snaps[-1] | ((snaps[-1] << sz) & mask_all))
    if not (snaps[-1] >> q) & 1:
        return None
    chosen: set[int] = set()
    for i in range(len(sizes), 0, -1):
        if (snaps[i - 1] >> q) & 1:
            continue
        chosen.add(i - 1)
        q -= sizes[i - 1]
    return chosen


def nearest_attainable(sizes: list[int], target: int) -> tuple[int, set[int]]:
    """Nearest attainable sum to target (exact first; ties -> higher)."""
    total = sum(sizes)
    for d in range(0, total + 1):
        cands = [target] if d == 0 else [target + d, target - d]
        for cand in cands:
            if cand < 0 or cand > total:
                continue
            got = subset_for_quota(sizes, cand)
            if got is not None:
                return cand, got
    raise AssertionError("no attainable sum (unreachable)")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = outputs_dir()
    release = find_release_root(root)
    assert release is not None, "release_not_found"
    cls = cls_split_dirs(release)
    pool = list_files(cls["train"], CLS_IMG_EXTS)
    test = list_files(cls["test"], CLS_IMG_EXTS)
    assert len(pool) == 5000 and len(test) == 1000, "pool/test counts changed"

    dup = json.loads((out / "duplicate_report.json").read_text())
    cross_hashes = set()
    for g in dup.get("cat3_cross_cls_train_test_CRITICAL", {}).get("hashes_sample", []):
        cross_hashes.add(g)
    # full 7-hash set (sample holds all 7: verified count below)
    assert dup["cat3_cross_cls_train_test_CRITICAL"]["n_shared_hashes"] == len(cross_hashes) == 7

    # test-side paths per contaminated hash (for exclusion list + N=993)
    test_by_hash: dict[str, list[str]] = defaultdict(list)
    test_hash_of: dict[str, str] = {}
    for p in test:
        h = sha256_file(p)
        test_by_hash[h].append(str(p.relative_to(root)))
        test_hash_of[str(p.relative_to(root))] = h
    train_by_hash: dict[str, list[str]] = defaultdict(list)
    for p in pool:
        train_by_hash[sha256_file(p)].append(str(p.relative_to(root)))

    sens_excluded = sorted({rp for h in cross_hashes for rp in test_by_hash[h]})
    sens_kept = sorted(set(test_hash_of) - set(sens_excluded))
    assert len(sens_kept) == 993, f"sensitivity N={len(sens_kept)}"
    write_json(
        "cross_split_exclusion_list.json",
        {
            "n_contaminated_hashes": 7,
            "hashes": sorted(cross_hashes),
            "train_paths": sorted([rp for h in cross_hashes for rp in train_by_hash[h]]),
            "excluded_test_paths": sens_excluded,
            "primary_test_n": 1000,
            "sensitivity_test_n": 993,
            "rule": "Primary N=1000 stays principal (official release). N=993 is sensitivity only.",
        },
    )

    # units (SHA-atomic) over pool
    units: dict[str, dict] = {}
    for h, paths in sorted(train_by_hash.items()):
        classes = {CLASS_ALIAS[Path(p).parent.name] for p in paths}
        assert len(classes) == 1, f"multi-class SHA unit {h[:12]}"
        parsed = [parse(Path(p).stem) for p in paths]
        assert all(c == next(iter(classes)) for c, _ in parsed), "folder/filename class mismatch"
        planes = sorted({v for _, v in parsed})
        assert all(v in VIEWS for v in planes), f"unknown plane code {h[:12]}"
        units[h] = {
            "files": sorted(paths),
            "n": len(paths),
            "class": next(iter(classes)),
            "planes": planes,
            "stratum": f"{next(iter(classes))}x{planes[0]}"
            if len(planes) == 1
            else next(iter(classes)),
            "contaminated": h in cross_hashes,
        }

    # R2: contaminated units -> train
    train_files: list[str] = []
    for h, u in units.items():
        if u["contaminated"]:
            train_files += u["files"]
    n_pre = len(train_files)
    assert n_pre == 9, f"contaminated train files={n_pre}"

    # R3: quotas per stratum over remaining units (largest remainder, exact)
    strata: dict[str, list[str]] = defaultdict(list)
    for h, u in units.items():
        if not u["contaminated"]:
            strata[u["stratum"]].append(h)
    for s in strata:
        strata[s].sort(key=lambda h: order_key(SEED, h))
    # file counts per stratum (clean units only)
    stratum_files = {s: sum(units[h]["n"] for h in hs) for s, hs in strata.items()}
    total_clean = sum(stratum_files.values())
    need = N_TRAIN - n_pre  # clean files to place in train
    quotas: dict[str, int] = {}
    fracs = []
    for s, f in stratum_files.items():
        q = need * f / total_clean
        quotas[s] = int(q)
        fracs.append((q - int(q), s))
    leftover = need - sum(quotas.values())
    for _, s in sorted(fracs, reverse=True)[:leftover]:
        quotas[s] += 1
    assert sum(quotas.values()) == need

    val_files: list[str] = []
    adjustments: dict[str, int] = {}
    carry = 0  # quota residual compensated by later (larger) strata
    for s in sorted(strata, key=lambda k: sum(units[h]["n"] for h in strata[k])):
        hs = strata[s]
        sizes = [units[h]["n"] for h in hs]
        target = quotas[s] + carry
        chosen_sum, chosen = nearest_attainable(sizes, target)
        adjustments[s] = chosen_sum - quotas[s]
        carry = target - chosen_sum  # residual for later (larger) strata
        for i in chosen:
            train_files += units[hs[i]]["files"]
        chosen_set = chosen
        for i, h in enumerate(hs):
            if i not in chosen_set:
                val_files += units[h]["files"]
    assert carry == 0, f"uncompensated quota residual {carry}"
    assert len(train_files) == N_TRAIN and len(val_files) == N_VAL
    assert not (set(train_files) & set(val_files))
    # R2 audit: no contaminated byte in validation
    val_hashes = {sha256_file(root / p) for p in val_files}
    assert not (val_hashes & cross_hashes), "contaminated byte in validation"

    train_files.sort()
    val_files.sort()
    manifest_rows = []
    split_of = dict.fromkeys(train_files, "train")
    split_of.update(dict.fromkeys(val_files, "val"))
    for p in sorted([str(x.relative_to(root)) for x in pool]):
        h = sha256_file(root / p)
        manifest_rows.append(
            {
                "path": p,
                "sha256": h,
                "population": "official_train_pool",
                "class": CLASS_ALIAS[Path(p).parent.name],
                "cross_split_contaminated": int(h in cross_hashes),
                "project_split": split_of[p],
                "sensitivity": "",
            }
        )
    for p in sorted([str(x.relative_to(root)) for x in test]):
        h = test_hash_of[p]
        manifest_rows.append(
            {
                "path": p,
                "sha256": h,
                "population": "official_test_locked",
                "class": CLASS_ALIAS[Path(p).parent.name],
                "cross_split_contaminated": int(h in cross_hashes),
                "project_split": "test_primary",
                "sensitivity": "excluded" if p in sens_excluded else "kept",
            }
        )
    with open(out / "project_manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    def _h(obj) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

    split = {
        "seed": SEED,
        "policy": "4000/1000 over 5000 pool; SHA-atomic units; "
        "class x plane strata (class-only fallback); "
        "contaminated units pre-assigned to train (R2); "
        "exact subset-sum DP per stratum, residual carried",
        "stratum_quotas": quotas,
        "atomicity_adjustments": adjustments,
        "train": train_files,
        "val": val_files,
        "test_primary": sorted(test_hash_of),
        "test_sensitivity": sens_kept,
        "hashes": {
            "train": _h(train_files),
            "val": _h(val_files),
            "manifest": _h([r["sha256"] for r in manifest_rows]),
        },
    }
    (out / "project_split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    (out / "project_split_report.md").write_text(
        "# Project split (frozen)\n\n"
        f"- train={len(train_files)} sha={split['hashes']['train']}\n"
        f"- val={len(val_files)} sha={split['hashes']['val']}\n"
        f"- test_primary=1000 LOCKED (contamination disclosed)\n"
        f"- test_sensitivity=993 LOCKED\n"
        f"- contaminated train files retained+flagged: {n_pre} (in train, never val)\n"
        f"- manifest sha={split['hashes']['manifest']}\n",
        encoding="utf-8",
    )
    print(
        f"split frozen: train={len(train_files)} val={len(val_files)} "
        f"sens=993 manifest={split['hashes']['manifest'][:12]}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
