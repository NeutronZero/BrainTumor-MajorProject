"""Shared PBA helpers: frozen checkpoint/T/taus loading, per-case inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.classification.models import build_classifier  # noqa: E402
from brain_tumor.data.brisc import load_manifest  # noqa: E402
from brain_tumor.preprocessing.pipeline import build_cls_transform  # noqa: E402

CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]


def load_frozen(root: Path):
    cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    clf = build_classifier("convnext_tiny")
    ckpt = torch.load(
        root / "checkpoints" / "CLS-001" / "best.pt", map_location="cpu", weights_only=False
    )  # own file
    clf.load_state_dict(ckpt["state"])
    clf.eval()
    return cal, manifest, clf


def infer_split(
    root: Path,
    clf,
    manifest,
    which: str,
    T: float,
    t1: float,
    t2: float,
    excl_test: set[str],
    tf=None,
):
    """Deterministic per-case inference. which: train/val/test(+sens flag).

    No decisions, no selection — pure recomputation of frozen artifacts."""
    from PIL import Image

    tf = tf or build_cls_transform(False)
    if which in ("train", "val"):
        rows = [
            r
            for r in manifest
            if r["population"] == "official_train_pool" and r["project_split"] == which
        ]
    else:
        rows = [r for r in manifest if r["population"] == "official_test_locked"]
    cases = []
    with torch.no_grad():
        for i, r in enumerate(rows):
            img = tf(Image.open(root / r["path"]).convert("RGB")).unsqueeze(0)
            probs = torch.softmax(clf(img) / T, dim=1)[0]
            top2 = probs.topk(2).values
            conf, margin = float(top2[0]), float(top2[0] - top2[1])
            pred = CLASSES[int(probs.argmax())]
            cases.append(
                {
                    "path": r["path"],
                    "split": which,
                    "true": r["class"],
                    "pred": pred,
                    "probs": [round(float(p), 6) for p in probs],
                    "confidence": round(conf, 6),
                    "margin": round(margin, 6),
                    "certain": bool(conf >= t1 and margin >= t2),
                    "contaminated": bool(r["path"] in excl_test),
                }
            )
            if i % 500 == 0 and i:
                print(f"infer {which} {i}/{len(rows)}", flush=True)
    return cases


def load_excl(root: Path) -> set[str]:
    excl = json.loads(
        (root / "outputs" / "data_gate_0" / "cross_split_exclusion_list.json").read_text()
    )
    return {p.replace("\\", "/") for p in excl["excluded_test_paths"]}


def img_chars(root: Path, relpath: str) -> dict:
    from PIL import Image

    with Image.open(root / relpath) as im:
        w, h = im.size
    return {"w": w, "h": h, "aspect": round(w / h, 3), "mp": round(w * h / 1e6, 3)}
