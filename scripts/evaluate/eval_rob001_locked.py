"""eval_rob001_locked.py — ROB-001 locked-test evaluation (PI-authorized).

Segmentation ONLY. Mirrors the SEG-001 locked protocol in
scripts/evaluate/evaluate_locked.py exactly (same 860 pairs, same frozen
seg pipeline + norm, threshold 0.5, same extract(), same _seg_block, same
sha-based N=853 sensitivity, same integrated-mode disagreement definition).

Test discipline:
  - Refuses unless test_set_locked is true.
  - NO classification inference: disagreement/system-states reuse FROZEN
    CLS-001 records (outputs/PBA-001/per_case_test.json, verified identical
    to the locked eval: 5 errors / 4 uncertain). Only ROB-001 UNet touches
    test pixels, including the live path for unpaired confident-tumor stems.
  - Writes outputs/ROB-001/locked_eval.json + freeze record. Never tunes.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.localization.extract import extract  # noqa: E402
from brain_tumor.preprocessing.pipeline import build_seg_pair_transform  # noqa: E402
from brain_tumor.segmentation.unet import UNet  # noqa: E402

BATCH = 16


def _open(p: Path, mode: str):
    from PIL import Image

    return Image.open(p).convert(mode)


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _file_sha(p: Path) -> str:
    return _sha(p)


def _seg_block(cases: list[dict], tag: str) -> dict:
    d = np.array([c["dice"] for c in cases])
    return {
        "tag": tag,
        "n": len(cases),
        "mean_dice": round(float(d.mean()), 6),
        "median_dice": round(float(np.median(d)), 6),
        "p10_dice": round(float(np.percentile(d, 10)), 6),
        "min_dice": round(float(d.min()), 6),
        "mean_iou": round(float(np.mean([c["iou"] for c in cases])), 6),
        "empty_pred": sum(1 for c in cases if c["empty_pred"]),
        "multi_component": sum(1 for c in cases if "multiple_components" in c["warnings"]),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    lock = yaml.safe_load(open(root / "configs" / "deployment" / "test_lock.yaml"))
    if not lock.get("test_set_locked", False):
        print("REFUSED: test_set_locked is not true. Unlock is a PI decision.")
        return 2
    out = root / "outputs" / "ROB-001"

    # ---- freeze record ----
    rob_cfg = root / "configs" / "experiment" / "ROB-001.yaml"
    freeze = {
        "experiment": "ROB-001",
        "checkpoint_sha256": _file_sha(root / "checkpoints" / "ROB-001" / "best.pt"),
        "config_sha256": _file_sha(rob_cfg),
        "config_matches_kernel_run": _file_sha(rob_cfg)
        == json.loads((out / "metrics.json").read_text())["config_sha256"],
        "pipeline": "src/brain_tumor/preprocessing/pipeline.py",
        "pipeline_sha256": _file_sha(
            root / "src" / "brain_tumor" / "preprocessing" / "pipeline.py"
        ),
        "localization": "configs/segmentation/localization.yaml",
        "localization_sha256": _file_sha(root / "configs" / "segmentation" / "localization.yaml"),
        "train_noise_sigma": 0.05,
        "threshold": 0.5,
        "morphology": "none (same as SEG-001)",
        "norm": json.loads((out / "metrics.json").read_text())["norm"],
        "val_selected": {
            "val_dice": json.loads((out / "metrics.json").read_text())["val_dice"],
            "best_epoch": json.loads((out / "metrics.json").read_text())["best_epoch"],
        },
    }
    assert freeze["config_matches_kernel_run"], "config changed since kernel run — aborting"
    (out / "freeze.json").write_text(json.dumps(freeze, indent=1))

    excl = json.loads(
        (root / "outputs" / "data_gate_0" / "cross_split_exclusion_list.json").read_text()
    )
    cont_hashes = set(excl["hashes"])

    seg_model = UNet()
    seg_model.load_state_dict(
        torch.load(
            root / "checkpoints" / "ROB-001" / "best.pt", map_location="cpu", weights_only=False
        )["state"]
    )
    seg_model.eval()
    seg_norm = freeze["norm"]
    seg_tf = build_seg_pair_transform(False, mean=seg_norm["mean"], std=seg_norm["std"])

    release = root / "data" / "brisc2025"
    pairs = []
    for img_p in sorted((release / "segmentation_task" / "test" / "images").glob("*.jpg")):
        m_p = release / "segmentation_task" / "test" / "masks" / (img_p.stem + ".png")
        assert m_p.exists(), f"missing test mask {img_p.stem}"
        pairs.append((img_p, m_p))
    assert len(pairs) == 860, "seg-test membership changed — aborting"

    seg_cases = []
    with torch.no_grad():
        ib, gb, meta = [], [], []

        def flush():
            if not ib:
                return
            prob = torch.sigmoid(seg_model(torch.stack(ib))).numpy()[:, 0]
            for pr, g, (img_p, m_p) in zip(prob, gb, meta):
                binp, bing = (pr > 0.5), (g > 0.5)
                inter = float((binp & bing).sum())
                dice = (2 * inter + 1e-6) / (float(binp.sum()) + float(bing.sum()) + 1e-6)
                iou = (inter + 1e-6) / (float((binp | bing).sum()) + 1e-6)
                ow, oh = _open(img_p, "RGB").size
                loc, w = extract(pr, (oh, ow))
                seg_cases.append(
                    {
                        "stem": img_p.stem,
                        "dice": dice,
                        "iou": iou,
                        "empty_pred": loc.area_pixels == 0,
                        "area": loc.area_pixels,
                        "bbox": loc.bbox,
                        "centroid": loc.centroid,
                        "warnings": w,
                        "contaminated": bool(_sha(img_p) in cont_hashes),
                    }
                )
            # meta holds paths; cleared by caller

        for i, (img_p, m_p) in enumerate(pairs):
            img, gt = seg_tf(_open(img_p, "L"), _open(m_p, "L"))
            ib.append(img)
            gb.append(gt[0].numpy())
            meta.append((img_p, m_p))
            if len(ib) == BATCH or i == len(pairs) - 1:
                flush()
                ib, gb, meta = [], [], []
            if i % 200 == 0 and i:
                print(f"seg {i}/860", flush=True)

    seg_primary = _seg_block(seg_cases, "rob001 seg-test N=860")
    seg_sens = _seg_block([c for c in seg_cases if not c["contaminated"]], "rob001 seg-test N=853")
    assert seg_sens["n"] == 853, f"sensitivity N={seg_sens['n']} != 853 — aborting"

    # ---- disagreement via FROZEN CLS-001 records (no CLS inference) ----
    cls_records = json.loads((root / "outputs" / "PBA-001" / "per_case_test.json").read_text())
    assert len(cls_records) == 1000
    assert sum(1 for c in cls_records if c["pred"] != c["true"]) == 5
    assert sum(1 for c in cls_records if not c["certain"]) == 4
    by_stem_seg = {c["stem"]: c for c in seg_cases}
    live_cache: dict[str, bool] = {}

    def _empty_for(path: str, stem: str) -> bool:
        if stem in by_stem_seg:
            return by_stem_seg[stem]["empty_pred"]
        if stem in live_cache:
            return live_cache[stem]
        pil_gray = _open(root / path, "L")
        ow, oh = _open(root / path, "RGB").size
        with torch.no_grad():
            img_t = seg_tf(pil_gray, pil_gray)[0]
            prob = torch.sigmoid(seg_model(img_t.unsqueeze(0)))[0, 0].numpy()
        loc, _ = extract(prob, (oh, ow))
        live_cache[stem] = loc.area_pixels == 0
        return live_cache[stem]

    dis, dis_by_cls, n_conf_tumor = 0, {}, 0
    sys_states: dict[str, int] = {}
    for c in cls_records:
        stem = Path(c["path"]).stem
        tumor = c["pred"] != "notumor"
        if tumor and c["certain"]:
            n_conf_tumor += 1
            if _empty_for(c["path"], stem):
                dis += 1
                dis_by_cls[c["pred"]] = dis_by_cls.get(c["pred"], 0) + 1
                st = "tumor_unlocalized"
            else:
                st = "tumor_localized"
        elif not tumor:
            st = "healthy" if c["certain"] else "uncertain"
        else:
            st = "uncertain"
        sys_states[st] = sys_states.get(st, 0) + 1
    disagreement = {
        "rate": dis / n_conf_tumor if n_conf_tumor else 0.0,
        "count": dis,
        "n_confident_tumor": n_conf_tumor,
        "per_class": dis_by_cls,
        "exceeds_5pct": (dis / n_conf_tumor > 0.05) if n_conf_tumor else False,
    }

    # ---- per-class dice (both populations) ----
    tmap = {"gl": "glioma", "me": "meningioma", "pi": "pituitary"}

    def by_class(cases):
        acc: dict[str, list] = {}
        for c in cases:
            acc.setdefault(tmap.get(c["stem"].split("_")[3], "?"), []).append(c["dice"])
        return {k: {"n": len(v), "mean": round(float(np.mean(v)), 6)} for k, v in acc.items()}

    # ---- deltas vs SEG-001 locked ----
    locked = json.loads((root / "outputs" / "test_evaluation_7b860dca72ea.json").read_text())
    s_p, s_s = locked["seg_primary"], locked["seg_sensitivity"]
    s_dis = locked["disagreement_confident_tumor_empty_seg"]

    def dseg(a, b):
        return {
            k: round(a[k] - b[k], 6) for k in ("mean_dice", "median_dice", "p10_dice", "mean_iou")
        }

    payload = {
        "experiment": "ROB-001 locked evaluation (segmentation only)",
        "freeze": freeze,
        "seg_primary": {**seg_primary, "by_class": by_class(seg_cases)},
        "seg_sensitivity": {
            **seg_sens,
            "by_class": by_class([c for c in seg_cases if not c["contaminated"]]),
        },
        "seg_delta_vs_SEG001": {
            "primary": dseg(seg_primary, s_p),
            "sensitivity": dseg(seg_sens, s_s),
            "empty_pred": seg_primary["empty_pred"] - s_p["empty_pred"],
            "multi_component": seg_primary["multi_component"] - s_p["multi_component"],
        },
        "system_state_distribution": sys_states,
        "disagreement_confident_tumor_empty_seg": disagreement,
        "disagreement_delta_vs_SEG001": {
            "count": disagreement["count"] - s_dis["count"],
            "rate": round(disagreement["rate"] - s_dis["rate"], 6),
        },
        "cls_source": "frozen CLS-001 records (no new classification inference)",
        "test_lock": "ROB-001 seg-test contact exactly once; no tuning permitted after this point",
    }
    (out / "locked_eval.json").write_text(json.dumps(payload, indent=1))
    print(
        json.dumps(
            {
                "seg_primary": payload["seg_primary"],
                "seg_sensitivity": {
                    k: payload["seg_sensitivity"][k]
                    for k in ("n", "mean_dice", "median_dice", "mean_iou")
                },
                "delta": payload["seg_delta_vs_SEG001"],
                "disagreement": disagreement,
                "system_states": sys_states,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
