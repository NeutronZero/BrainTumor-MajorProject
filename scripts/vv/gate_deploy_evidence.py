"""V&V-001 Gates 4-5: deployment-path + evidence-integrity verification.

Gate 4: CPU leg runs live here (service load + synthetic inference +
determinism); CUDA leg cross-checks RECORDED T4 evidence (no new kernel,
no optimization): determinism flags, FP16 floors, bench definitions.
Gate 5: checkpoint hashes + locked-vs-recorded number spot checks.
Writes results to stdout as JSON (caller archives to outputs/VV/).
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import time
from pathlib import Path

import numpy as np


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from PIL import Image
    from brain_tumor.inference.service import InferenceService

    checks: dict[str, dict] = {}

    def check(name, ok, detail=""):
        checks[name] = {"pass": bool(ok), "detail": str(detail)}

    # Gate 4 CPU leg
    t0 = time.perf_counter()
    svc = InferenceService.from_registry()
    check("cpu_load", svc.clf is not None and svc.seg is not None,
          f"{time.perf_counter() - t0:.1f}s")
    rng = np.random.RandomState(31)
    buf = io.BytesIO()
    Image.fromarray((rng.rand(256, 256) * 255).astype("uint8")).save(buf, format="PNG")
    buf.seek(0)
    img = Image.open(buf)
    t = time.perf_counter()
    r1 = svc.analyze(img).model_dump()
    dt = time.perf_counter() - t
    r2 = svc.analyze(img).model_dump()
    check("cpu_inference", r1["system_state"] in
          ("tumor_localized", "tumor_unlocalized", "healthy", "uncertain", "degraded"),
          f"{dt:.2f}s -> {r1['system_state']}")
    check("cpu_determinism", r1 == r2)

    # Gate 4 CUDA leg: recorded T4 evidence must still support frozen decisions
    fp16 = json.loads((root / "outputs" / "SYSINT" / "fp16_check.json").read_text())
    onnx = json.loads((root / "outputs" / "SYSINT" / "onnx_check.json").read_text())
    bk8 = json.loads((root / "outputs" / "SYSINT" / "batchk8_check.json").read_text())
    check("t4_fp16_floors", fp16["floors"] == {"macro_f1_pass": True, "mean_dice_pass": True}
          and fp16["fp16_deterministic_repeat"] is True)
    check("t4_onnx_split", onnx["correctness"]["f1_floor_pass"]
          and onnx["correctness"]["dice_floor_pass"]
          and onnx["deployment"]["latency_s"]["analyze"]["median"] >= 0.034)
    check("t4_bk8_rejected", bk8["equivalence_val_N1000"]["flag_mismatches"] == 0
          and bk8["latency_s"]["consistency_batched"]["median"]
          > bk8["latency_s"]["consistency_seq"]["median"])

    # Gate 5: hashes + locked numbers
    for p, pre in (("checkpoints/CLS-001/best.pt", "0fa58033"),
                   ("checkpoints/SEG-001/best.pt", "ce29df5e"),
                   ("checkpoints/ROB-001/best.pt", "36091d63")):
        check(f"hash:{p}", sha(root / p).startswith(pre), sha(root / p)[:16])
    ev = json.loads((root / "outputs" / "test_evaluation_7b860dca72ea.json").read_text())
    check("locked_cls", (ev["cls_primary"]["accuracy"], ev["cls_primary"]["macro_f1"],
                         ev["cls_primary"]["uncertain"]) == (0.995, 0.995167, 4))
    check("locked_seg", (ev["seg_primary"]["mean_dice"], ev["seg_primary"]["empty_pred"],
                         ev["disagreement_confident_tumor_empty_seg"]["count"]) == (0.861659, 10, 10))
    check("locked_sens", (ev["cls_sensitivity"]["accuracy"], ev["seg_sensitivity"]["mean_dice"])
          == (0.994965, 0.860962))
    u = json.loads((root / "outputs" / "UNC-001" / "unc001_locked.json").read_text())
    check("locked_unc", (u["detection"]["detection_rate"],
                         u["burden"]["false_alert_rate"]) == (0.925926, 0.033166))
    r = json.loads((root / "outputs" / "ROB-001" / "locked_eval.json").read_text())
    check("locked_rob", (r["seg_primary"]["mean_dice"],
                         r["disagreement_confident_tumor_empty_seg"]["count"]) == (0.820744, 9))

    failed = [k for k, v in checks.items() if not v["pass"]]
    print(json.dumps({"failed": failed, "checks": checks}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
