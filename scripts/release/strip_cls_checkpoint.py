"""Derive the state-only CLS-001 production checkpoint (C3, Option A).

Reads checkpoints/CLS-001/best.pt (frozen, never modified), writes
checkpoints/CLS-001/best_stateonly.pt containing model state + provenance
(epoch, val_metric, lineage) and NO optimizer/scheduler/scaler/RNG tensors.
Then proves equivalence: identical state dicts, identical inference on a
synthetic battery, identical parameter count. The service loads ["state"]
from either file. Locked evaluation remains attributed to the original
checkpoint (documented lineage, not a silent swap).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch


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
    from brain_tumor.classification.models import build_classifier
    from brain_tumor.preprocessing.pipeline import build_cls_transform

    src = root / "checkpoints" / "CLS-001" / "best.pt"
    dst = root / "checkpoints" / "CLS-001" / "best_stateonly.pt"
    ck = torch.load(src, map_location="cpu", weights_only=False)
    assert set(ck) >= {"state", "epoch", "val_metric"}, sorted(ck)
    stripped = {
        "state": ck["state"],
        "epoch": ck["epoch"],
        "val_metric": ck["val_metric"],
        "lineage": {
            "derived_from_sha256": sha(src),
            "method": "key removal only (optimizer/scheduler/scaler/rng dropped)",
            "locked_eval_attribution": "original best.pt (unchanged)",
        },
    }
    torch.save(stripped, dst)

    # Equivalence proof.
    a = build_classifier("convnext_tiny")
    a.load_state_dict(torch.load(src, map_location="cpu", weights_only=False)["state"])
    b = build_classifier("convnext_tiny")
    b.load_state_dict(torch.load(dst, map_location="cpu", weights_only=False)["state"])
    sa, sb = a.state_dict(), b.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert (
        sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters()) == 27823204
    )
    tf = build_cls_transform(False)
    cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
    a.eval()
    b.eval()
    for seed, size in ((31, (256, 256)), (5, (512, 512)), (7, (300, 400))):
        rng = np.random.RandomState(seed)
        w, h = size
        img = Image.fromarray((rng.rand(h, w) * 255).astype("uint8"))
        x = tf(img.convert("RGB")).unsqueeze(0)
        with torch.no_grad():
            pa = torch.softmax(a(x) / cal["temperature"], dim=1)[0]
            pb = torch.softmax(b(x) / cal["temperature"], dim=1)[0]
        assert torch.equal(pa, pb), (seed, size)
        assert pa.argmax().item() == pb.argmax().item()
    print(
        json.dumps(
            {
                "src_sha": sha(src)[:16],
                "dst_sha": sha(dst)[:16],
                "src_bytes": src.stat().st_size,
                "dst_bytes": dst.stat().st_size,
                "state_tensors": len(sa),
                "inference_identical_3_cases": True,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
