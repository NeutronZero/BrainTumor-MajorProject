"""P2 API schema regression: /analyze payload contract + all endpoints.

Asserts existing fields unchanged, quality present, derived fields derived,
envelopes intact, all nine endpoints live, service/API equivalence.
Synthetic input only. Writes outputs/DEMO/acceptance_api.json.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

CANONICAL = ("predicted_class", "probabilities", "confidence",
             "classification_state", "segmentation_state", "localization",
             "warnings", "system_state")
OBSERVER_KEYS = ("quality", "consistency", "reliability")


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
    rng = np.random.RandomState(31)
    buf = io.BytesIO()
    Image.fromarray((rng.rand(256, 256) * 255).astype("uint8")).save(buf, format="PNG")
    raw = buf.getvalue()
    client = TestClient(api_main.app)
    checks: dict[str, dict] = {}

    def check(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": str(detail)}

    a = client.post("/analyze", files={"file": ("t.png", raw, "image/png")}).json()
    check("analyze_has_canonical", all(k in a for k in CANONICAL))
    check("analyze_has_quality", "quality" in a and a["quality"]["verdict"] == "accept")
    check("analyze_has_observers", all(k in a for k in OBSERVER_KEYS))
    check("derived_state_coherent",
          (a["system_state"] == "healthy") == (a["predicted_class"] == "notumor")
          or a["system_state"] in ("uncertain", "degraded"),
          (a["predicted_class"], a["system_state"]))
    svc = InferenceService.from_registry()
    s = svc.analyze(Image.open(io.BytesIO(raw))).model_dump()

    def norm(d):
        loc = dict(d["localization"])
        if loc.get("bbox") is not None:
            loc["bbox"] = tuple(loc["bbox"])
        return {k: (loc if k == "localization" else d[k]) for k in CANONICAL}
    check("service_api_equivalence", norm(s) == norm(a))
    for ep in ("classify", "segment", "localize", "quality", "consistency",
               "reliability", "explain"):
        r = client.post(f"/{ep}", files={"file": ("t.png", raw, "image/png")})
        check(f"endpoint_{ep}", r.status_code == 200, r.status_code)
    check("env_415", client.post("/analyze", files={"file": ("t", raw, "text/plain")}).status_code == 415)
    big = client.post("/analyze", files={"file": ("t.png", b"x" * (11 * 1024 * 1024), "image/png")})
    check("env_413", big.status_code == 413)
    bad = client.post("/analyze", files={"file": ("t.png", b"junk", "image/png")})
    check("env_422", bad.status_code == 422)
    check("health", client.get("/health").json()["status"] == "ok")

    failed = [k for k, v in checks.items() if not v["pass"]]
    (out / "acceptance_api.json").write_text(json.dumps(
        {"failed": failed, "checks": checks}, indent=1))
    print(json.dumps({"failed": failed}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
