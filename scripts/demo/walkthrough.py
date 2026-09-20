"""P2 deployment walkthrough (headless): start -> load image -> analyze ->
read model output -> read observers -> explain -> limitations -> stop.

Synthetic input only. Writes outputs/DEMO/walkthrough.json.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "app" / "api"))
    from PIL import Image
    from brain_tumor.inference.service import InferenceService
    import main as api_main
    from fastapi.testclient import TestClient

    out = root / "outputs" / "DEMO"
    out.mkdir(parents=True, exist_ok=True)
    log = []

    t0 = time.perf_counter()
    svc = InferenceService.from_registry()
    log.append({"step": "start_system", "classifier": svc.clf is not None,
                "segmenter": svc.segmentation_available,
                "load_s": round(time.perf_counter() - t0, 2)})
    rng = np.random.RandomState(0)
    yy, xx = np.mgrid[0:256, 0:256]
    blob = np.where(((xx - 150) ** 2 + (yy - 128) ** 2) < 40**2, 220, 30).astype(float)
    raw = io.BytesIO()
    Image.fromarray(np.clip(blob + rng.rand(256, 256) * 15, 0, 255).astype("uint8")).save(
        raw, format="PNG")
    log.append({"step": "load_image", "bytes": len(raw.getvalue())})
    img = Image.open(io.BytesIO(raw.getvalue()))
    r = svc.analyze(img).model_dump()
    log.append({"step": "read_model_output", "class": r["predicted_class"],
                "confidence": r["confidence"], "state": r["classification_state"]})
    log.append({"step": "read_observers",
                "quality": svc.quality(raw.getvalue())["verdict"],
                "agreement": svc.consistency(img)["agreement_fraction"],
                "reliability": svc.reliability(img)["reliability"]["summary"]})
    e = svc.explain(img)
    log.append({"step": "inspect_explanation",
                "same_prediction": e["predicted_class"] == r["predicted_class"],
                "has_report": "report_text" in e, "has_overlay": bool(
                    e["gradcam"].get("overlay_png_b64"))})
    log.append({"step": "interpret_limitations",
                "disclaimer": e["disclaimer"][:60], "system_state": r["system_state"]})
    client = TestClient(api_main.app)
    log.append({"step": "api_health", "status": client.get("/health").json()["status"]})
    log.append({"step": "stop_system", "note": "no persistent state; process exit"})
    (out / "walkthrough.json").write_text(json.dumps({"steps": log}, indent=1))
    print(json.dumps([s["step"] for s in log]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
