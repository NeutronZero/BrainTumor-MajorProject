"""Demo matrix: 7 deterministic synthetic cases exercising system contracts.

Demonstrates contracts, never accuracy. Each case pins a fixture (fixed
seed/shape) and asserts contract-level outcomes (states, codes, branches).
A failure means the FIXTURE no longer exercises its path (document and
replace the fixture) — never a reason to tune the model. Synthetic only.
Writes outputs/DEMO/demo_matrix.json.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.inference.service import InferenceService  # noqa: E402


def _png(a: np.ndarray) -> tuple:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.clip(a, 0, 255).astype("uint8")).save(buf, format="PNG")
    return Image.open(io.BytesIO(buf.getvalue())), buf.getvalue()


def main() -> int:
    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[2]
    out = root / "outputs" / "DEMO"
    out.mkdir(parents=True, exist_ok=True)
    svc = InferenceService.from_registry()
    rng = np.random.RandomState(0)

    checker = (np.indices((256, 256)).sum(axis=0) % 2 * 255).astype(float)
    yy, xx = np.mgrid[0:256, 0:256]
    blob = np.where(((xx - 150) ** 2 + (yy - 128) ** 2) < 40**2, 220, 30).astype(float)
    blob += rng.rand(256, 256) * 15
    cases = {
        "healthy_like": (checker, {"predicted_class": "notumor"}),
        "tumor_like": (rng.rand(256, 256) * 255, {"predicted_class": "pituitary",
                                                 "classification_state": "confident"}),
        "low_quality": (np.zeros((256, 256)), {"quality": "reject"}),
        "seg_empty": (np.zeros((256, 256)), {"predicted_class": "glioma",
                                             "segmentation_state": "empty"}),
        "seg_localized": (blob, {"predicted_class": "meningioma",
                                 "system_state": "tumor_localized"}),
        "uncertain": (np.full((256, 256), 128.0), {"classification_state": "uncertain"}),
    }
    rows = []
    for name, (arr, exp) in cases.items():
        img, raw = _png(arr)
        r = svc.analyze(img).model_dump()
        q = svc.quality(raw)["verdict"]
        got = {"predicted_class": r["predicted_class"],
               "classification_state": r["classification_state"],
               "system_state": r["system_state"],
               "segmentation_state": r["segmentation_state"], "quality": q}
        exp_full = dict(exp)
        ok = all(got[k] == v for k, v in exp_full.items())
        rows.append({"case": name, "expected": exp_full, "observed": got,
                     "contract_holds": ok})
        print(f"{name}: {'OK' if ok else 'FIXTURE-DRIFT'} {got}", flush=True)

    # degraded path (missing segmenter): contract-level degraded behavior
    noseg = InferenceService(
        classifier_ckpt=root / "checkpoints" / "CLS-001" / "best.pt",
        segmenter_ckpt=root / "checkpoints" / "MISSING" / "best.pt")
    img, _ = _png(rng.rand(256, 256) * 255)
    rd = noseg.analyze(img).model_dump()
    rows.append({"case": "degraded_no_segmenter",
                 "expected": {"system_state": "degraded"},
                 "observed": {"system_state": rd["system_state"]},
                 "contract_holds": rd["system_state"] == "degraded"})
    (out / "demo_matrix.json").write_text(json.dumps({"cases": rows}, indent=1))
    bad = [c["case"] for c in rows if not c["contract_holds"]]
    print("drifted fixtures:", bad if bad else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
