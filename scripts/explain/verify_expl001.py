"""EXPL-001 evidence: implementation hashes + live observer-contract check.

Synthetic image only. Records target layer, module hashes, equivalence
(analyze vs explain outputs/state), determinism, and limitations.
Writes outputs/EXPL-001/expl001.json.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    from PIL import Image
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from brain_tumor.explain.gradcam import resolve_target_layer
    from brain_tumor.inference.service import InferenceService

    out = root / "outputs" / "EXPL-001"
    out.mkdir(parents=True, exist_ok=True)
    svc = InferenceService.from_registry()
    name, _ = resolve_target_layer(svc.clf)
    rng = np.random.RandomState(41)
    buf = io.BytesIO()
    Image.fromarray((rng.rand(256, 256) * 255).astype("uint8")).save(buf, format="PNG")
    buf.seek(0)
    img = Image.open(buf)
    a = svc.analyze(img).model_dump()
    e1, e2 = svc.explain(img), svc.explain(img)
    keys = ("predicted_class", "probabilities", "confidence", "classification_state",
            "segmentation_state", "localization", "warnings", "system_state")
    rep = {
        "target_layer": name,
        "module_hashes": {
            "explain/gradcam.py": sha(root / "src/brain_tumor/explain/gradcam.py")[:16],
            "explain/report.py": sha(root / "src/brain_tumor/explain/report.py")[:16],
            "inference/service.py": sha(root / "src/brain_tumor/inference/service.py")[:16],
        },
        "observer_equivalence": all(e1[k] == a[k] for k in keys),
        "deterministic_heatmap": (e1["gradcam"]["heatmap_png_b64"]
                                  == e2["gradcam"]["heatmap_png_b64"]),
        "report_disclaimer_present": "not a clinical" in e1["report_text"],
        "precision_note": "explanation forwards run FP32 on all devices (documented choice)",
        "limitations": ["heatmap is a contribution visualization, not proof of attention",
                        "no clinical-meaning claim without separate validation",
                        "focus_in_bbox is descriptive, not a detection metric"],
    }
    (out / "expl001.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    return 0 if (rep["observer_equivalence"] and rep["deterministic_heatmap"]) else 1


if __name__ == "__main__":
    sys.exit(main())
