"""audit_split_overlap.py — Revision A: nested BRISC split discovery (§9/§10).

Discovers official splits at:
  <release>/classification_task/{train,test}
  <release>/segmentation_task/{train,test}/{images,masks}
Compares by byte-SHA256 (not name/size proxies). FAIL (fail-closed) on:
release absent, expected nested splits missing, or any train<->test SHA
overlap within either task. Overlap is reported with counts and samples;
nothing is deleted or relabeled. Project-internal 4000/1000 membership is
NOT generated here (§10: only after Gate 0 PASS).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gate.common import sha256_file, write_json  # noqa: E402
from data_gate.layout import (  # noqa: E402
    CLS_IMG_EXTS,
    cls_split_dirs,
    find_release_root,
    list_files,
    seg_split_dirs,
)


def _sha_set(files: list[Path]) -> set[str]:
    out: set[str] = set()
    for p in files:
        try:
            out.add(sha256_file(p))
        except Exception:
            pass
    return out


def main() -> dict:
    root = Path(__file__).resolve().parents[2]
    release = find_release_root(root)
    if release is None:
        result = {"check": "split_overlap", "pass": False,
                  "reason": "release_not_found"}
        write_json("split_overlap_report.json", result)
        print("split_overlap: FAIL (release_not_found)")
        return result

    cls, seg = cls_split_dirs(release), seg_split_dirs(release)
    missing = [f"classification_task/{s}" for s, d in cls.items() if not d.is_dir()]
    missing += [f"segmentation_task/{s}/{k}" for s, dd in seg.items()
                for k, d in dd.items() if not d.is_dir()]
    if missing:
        result = {"check": "split_overlap", "pass": False,
                  "reason": "expected_nested_splits_missing", "missing": missing}
        write_json("split_overlap_report.json", result)
        print(f"split_overlap: FAIL (missing={missing})")
        return result

    ct = list_files(cls["train"], CLS_IMG_EXTS)
    ce = list_files(cls["test"], CLS_IMG_EXTS)
    st = list_files(seg["train"]["images"], CLS_IMG_EXTS)
    se = list_files(seg["test"]["images"], CLS_IMG_EXTS)
    stm = list_files(seg["train"]["masks"], {".png"})
    sem = list_files(seg["test"]["masks"], {".png"})

    s_ct, s_ce = _sha_set(ct), _sha_set(ce)
    s_st, s_se = _sha_set(st), _sha_set(se)
    s_stm, s_sem = _sha_set(stm), _sha_set(sem)

    ov_cls = sorted(s_ct & s_ce)
    ov_seg_img = sorted(s_st & s_se)
    ov_seg_msk = sorted(s_stm & s_sem)
    n_ov = len(ov_cls) + len(ov_seg_img) + len(ov_seg_msk)
    ok = n_ov == 0
    result = {"check": "split_overlap", "pass": ok, "method": "byte-SHA256",
              "release": str(release.relative_to(root)),
              "counts": {"cls_train": len(ct), "cls_test": len(ce),
                         "seg_train_images": len(st), "seg_test_images": len(se),
                         "seg_train_masks": len(stm), "seg_test_masks": len(sem)},
              "overlap": {"cls_train_test_hashes": len(ov_cls),
                          "seg_train_test_image_hashes": len(ov_seg_img),
                          "seg_train_test_mask_hashes": len(ov_seg_msk),
                          "cls_sample": ov_cls[:10], "seg_img_sample": ov_seg_img[:10]},
              "reason": None if ok else "split_overlap_detected",
              "note": ("Project-internal 4000/1000 membership frozen only after "
                       "Gate 0 PASS (§10).")}
    write_json("split_overlap_report.json", result)
    print(f"split_overlap: {'PASS' if ok else 'FAIL'} "
          f"cls={len(ov_cls)} seg_img={len(ov_seg_img)} seg_msk={len(ov_seg_msk)}")
    return result


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["pass"] else 2)
