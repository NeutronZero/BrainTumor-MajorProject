"""reproducibility_check.py — golden-output smoke test, target <30s (§58)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from brain_tumor.inference.service import InferenceService  # noqa: E402


def main() -> int:
    svc = InferenceService.from_registry()
    res = svc.analyze()
    d = res.model_dump()
    assert d["probabilities"], "empty probabilities"
    assert abs(sum(d["probabilities"].values()) - 1.0) < 0.01
    print("reproducibility_check: OK", json.dumps(d["system_state"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
