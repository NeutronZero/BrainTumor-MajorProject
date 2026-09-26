"""V&V-001 Gate 3: service == API == Streamlit payload on identical synthetic input.

Compares canonical BrainTumorResult fields across the three interfaces.
Writes results to stdout as JSON (caller archives to outputs/VV/).
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

FIELDS = (
    "predicted_class",
    "probabilities",
    "confidence",
    "classification_state",
    "segmentation_state",
    "localization",
    "warnings",
    "system_state",
)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "app" / "api"))
    sys.path.insert(0, str(root / "app" / "streamlit"))
    import app as streamlit_app
    import main as api_main
    from fastapi.testclient import TestClient
    from PIL import Image

    from brain_tumor.inference.service import InferenceService

    rng = np.random.RandomState(31)
    data = io.BytesIO()
    Image.fromarray((rng.rand(256, 256) * 255).astype("uint8")).save(data, format="PNG")
    raw = data.getvalue()
    img = Image.open(io.BytesIO(raw))

    svc = InferenceService.from_registry()
    a = svc.analyze(img).model_dump()
    client = TestClient(api_main.app)
    b = client.post(
        "/analyze",
        files={"file": ("t.png", raw, "image/png")},
        params={"consistency_probes": False},
    ).json()
    c = streamlit_app.build_payload(svc, raw)

    norm = lambda d: {
        k: (tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in ((k, d[k]) for k in FIELDS)
    }

    # localization dicts: tuples vs lists
    def flat(d):
        loc = dict(d["localization"])
        if loc.get("bbox") is not None:
            loc["bbox"] = tuple(loc["bbox"])
        return {k: (loc if k == "localization" else d[k]) for k in FIELDS}

    fa, fb, fc = flat(a), flat(b), flat({k: c[k] for k in FIELDS})
    rep = {
        "service_vs_api": fa == fb,
        "service_vs_streamlit": fa == fc,
        "system_state": fa["system_state"],
        "predicted": fa["predicted_class"],
    }
    print(json.dumps(rep, indent=1))
    return 0 if (rep["service_vs_api"] and rep["service_vs_streamlit"]) else 1


if __name__ == "__main__":
    sys.exit(main())
