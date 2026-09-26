"""auc_addendum.py — computes the missing ROC-AUC OvR for the frozen locked eval.

Root cause of null: roc_auc_score received string y_true with int `labels`.
This script changes NOTHING (same frozen checkpoint, same T, same test set,
no thresholds, no selection) and writes a sidecar addendum; the original
test_evaluation_<sha>.json is preserved byte-identical (one-way gate intact).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.classification.models import build_classifier  # noqa: E402
from brain_tumor.data.brisc import CLASS_TO_IDX, load_manifest  # noqa: E402
from brain_tumor.preprocessing.pipeline import build_cls_transform  # noqa: E402

CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    base = sorted(root.glob("outputs/test_evaluation_*.json"))
    base = [p for p in base if "addendum" not in p.name]
    assert len(base) == 1, f"expected one locked eval, found {base}"
    eval_path = base[0]
    payload = json.loads(eval_path.read_text())
    cal = payload["calibration"]
    T = cal["T"]

    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    test_rows = [r for r in manifest if r["population"] == "official_test_locked"]
    excl = json.loads(
        (root / "outputs" / "data_gate_0" / "cross_split_exclusion_list.json").read_text()
    )
    excl_test = {p.replace("\\", "/") for p in excl["excluded_test_paths"]}

    clf = build_classifier("convnext_tiny")
    ckpt = torch.load(
        root / "checkpoints" / "CLS-001" / "best.pt", map_location="cpu", weights_only=False
    )  # own file
    clf.load_state_dict(ckpt["state"])
    clf.eval()
    tf = build_cls_transform(False)

    from PIL import Image

    probs_all, y_all, keep = [], [], []
    with torch.no_grad():
        for i, r in enumerate(test_rows):
            img = tf(Image.open(root / r["path"]).convert("RGB")).unsqueeze(0)
            probs_all.append(torch.softmax(clf(img) / T, dim=1)[0].tolist())
            y_all.append(CLASS_TO_IDX[r["class"]])
            keep.append(r["path"] not in excl_test)
            if i % 250 == 0:
                print(f"auc {i}/1000", flush=True)

    import numpy as np
    from sklearn.metrics import roc_auc_score

    P, Y = np.array(probs_all), np.array(y_all)
    auc1000 = float(roc_auc_score(Y, P, multi_class="ovr"))
    mask = np.array(keep)
    auc993 = float(roc_auc_score(Y[mask], P[mask], multi_class="ovr"))
    add = {
        "addendum_to": eval_path.name,
        "reason": "roc_auc_ovr was null in locked eval (string/int label-type bug in reporting only)",
        "roc_auc_ovr_primary_N1000": round(auc1000, 6),
        "roc_auc_ovr_sensitivity_N993": round(auc993, 6),
        "inputs_unchanged": {
            "checkpoint": "checkpoints/CLS-001/best.pt",
            "T": T,
            "model_selection": "none",
        },
    }
    out = eval_path.with_name(eval_path.stem + "_addendum_auc.json")
    out.write_text(json.dumps(add, indent=1))
    with open(root / "outputs" / "test_evaluation_log.md", "a", encoding="utf-8") as f:
        f.write(
            f"\n- addendum {out.name} reason='AUC reporting fix, frozen inputs identical' "
            f"auc1000={auc1000:.4f} auc993={auc993:.4f}\n"
        )
    print(json.dumps(add, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
