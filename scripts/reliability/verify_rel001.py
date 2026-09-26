"""REL-001 evidence: rule doc, module hashes, synthetic combination matrix.

Synthetic images only; no locked-test contact. Writes
outputs/REL-001/rel001.json.
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
    from brain_tumor.inference.service import InferenceService

    out = root / "outputs" / "REL-001"
    out.mkdir(parents=True, exist_ok=True)
    svc = InferenceService.from_registry()

    def png(a):
        buf = io.BytesIO()
        Image.fromarray(a.astype("uint8")).save(buf, format="PNG")
        buf.seek(0)
        return Image.open(buf), buf.getvalue()

    rng = np.random.RandomState(31)
    matrix = {}
    cases = {
        "textured": rng.rand(256, 256) * 255,
        "checker_notumor": (np.indices((256, 256)).sum(axis=0) % 2 * 255).astype(float),
        "dark_empty_seg": np.zeros((256, 256)),
        "blank_reject": np.zeros((256, 256)),
    }
    for name, arr in cases.items():
        img, raw = png(arr)
        r = svc.reliability(img, raw=raw)
        matrix[name] = {
            "predicted": r["classification"]["class"],
            "state": r["classification"]["state"],
            "agreement": r["consistency"]["agreement"],
            "quality": r["quality"]["state"],
            "seg_nonempty": r["segmentation"]["nonempty"],
            "summary": r["reliability"]["summary"],
            "n_basis": len(r["reliability"]["basis"]),
        }
    rep = {
        "rule": "stable iff confident + agreement==1.0 + quality accept + "
        "(notumor OR nonempty+localized); else review. Fixed, untuned.",
        "module_hashes": {
            "reliability/engine.py": sha(root / "src/brain_tumor/reliability/engine.py")[:16],
            "inference/service.py": sha(root / "src/brain_tumor/inference/service.py")[:16],
        },
        "combination_matrix_synthetic": matrix,
        "observer_note": "reliability() cannot alter SB-1 outputs (pure fusion; "
        "equivalence asserted in tests/integration/test_reliability.py)",
        "limitations": [
            "descriptive summary, no validated statistical meaning",
            "no single reliability score constructed",
            "no clinical-reliability claim; clinical_meaning always false",
            "does not override predictions or system_state",
        ],
    }
    (out / "rel001.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(matrix, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
