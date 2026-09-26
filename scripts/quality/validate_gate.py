"""Validate the quality gate: false-rejects on clean val + corruption battery.

Development data only (project val N=1000; locked test never touched).
Corruptions are synthetic, generated in-memory. Reports false-rejection
rate and per-corruption detection rate. Thresholds are FIXED in gate.py;
this script only measures. Writes outputs/QUALITY/gate_validation.json.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.data.brisc import load_manifest  # noqa: E402
from brain_tumor.quality.gate import assess  # noqa: E402


def _png(a: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(a.astype("uint8")).save(buf, format="PNG")
    return buf.getvalue()


def corruptions():
    rng = np.random.RandomState(99)
    return {
        "blank_black": _png(np.zeros((256, 256))),
        "blank_white": _png(np.full((256, 256), 255)),
        "uniform_gray": _png(np.full((256, 256), 128)),
        "pure_noise": _png(rng.rand(256, 256) * 255),
        "tiny_32px": _png(rng.rand(32, 32) * 255),
        "stripe": _png(rng.rand(40, 800) * 255),
        "truncated": _png(rng.rand(256, 256) * 255)[:500],
        "text_bytes": b"this is not an image at all" * 10,
        "near_blank": _png(np.full((256, 256), 5)),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "QUALITY"
    out.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
    val = [
        r
        for r in manifest
        if r["population"] == "official_train_pool" and r["project_split"] == "val"
    ]
    assert len(val) == 1000 and not any(r["population"] == "official_test_locked" for r in val)

    rejects = []
    for i, r in enumerate(val):
        data = open(root / r["path"], "rb").read()
        a = assess(data)
        if a["verdict"] == "reject":
            rejects.append({"path": r["path"], "failed": a["failed"], "facts": a["facts"]})
        if (i + 1) % 500 == 0:
            print(f"clean-val {i + 1}/1000 rejects={len(rejects)}", flush=True)

    det = {}
    for name, data in corruptions().items():
        a = assess(data)
        det[name] = {"verdict": a["verdict"], "failed": a["failed"]}
        print(f"corruption {name}: {a['verdict']} {a['failed']}", flush=True)

    rep = {
        "clean_val_n": len(val),
        "false_rejects": len(rejects),
        "false_reject_rate": round(len(rejects) / len(val), 6),
        "false_reject_cases": rejects[:20],
        "corruption_detection": det,
        "note": "thresholds fixed in gate.py; val used for reporting only; no test contact",
    }
    (out / "gate_validation.json").write_text(json.dumps(rep, indent=1))
    print(
        json.dumps(
            {
                "false_rejects": len(rejects),
                "detected": sum(1 for v in det.values() if v["verdict"] == "reject"),
                "of": len(det),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
