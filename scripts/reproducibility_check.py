"""reproducibility_check.py — golden-output smoke test, target <30s.

Deterministic synthetic input (seed 31); asserts repeated inference gives
identical classification, confidence, and system state. No data, no network.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brain_tumor.inference.service import InferenceService  # noqa: E402


def main() -> int:
    from PIL import Image

    svc = InferenceService.from_registry()
    rng = np.random.RandomState(31)
    buf = io.BytesIO()
    Image.fromarray((rng.rand(256, 256) * 255).astype("uint8")).save(buf, format="PNG")
    raw = buf.getvalue()
    r1 = svc.analyze(Image.open(io.BytesIO(raw))).model_dump()
    r2 = svc.analyze(Image.open(io.BytesIO(raw))).model_dump()
    assert r1["predicted_class"] == r2["predicted_class"], "nondeterministic class"
    assert r1["confidence"] == r2["confidence"], "nondeterministic confidence"
    assert r1["system_state"] == r2["system_state"], "nondeterministic state"
    assert r1["probabilities"], "empty probabilities"
    assert abs(sum(r1["probabilities"].values()) - 1.0) < 0.01
    print(
        "reproducibility_check: OK",
        json.dumps({"class": r1["predicted_class"], "state": r1["system_state"]}),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
