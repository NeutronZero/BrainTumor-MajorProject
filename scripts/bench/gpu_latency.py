"""GPU latency reference (FP32) — deployment measurement, NOT a model experiment.

Frozen stack: CLS-001 + SEG-001 + frozen calibration + UNC-001 descriptive
signal. Synthetic inputs only (fixed seeds); no locked-test images, no data
dependency. Measures: cold start (import + load), warm per-component latency
(classify / segment / consistency K=8 / full analyze), peak GPU memory, and
run-to-run distributions (median/p95). Writes outputs/SYSINT/gpu_latency_fp32.json.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

REPEATS = 20
SEEDS = (101, 202, 303)


def _synthetic(seed: int, size: int = 512):
    from PIL import Image
    rng = np.random.RandomState(seed)
    return Image.fromarray((rng.rand(size, size) * 255).astype("uint8"))


def _stats(xs: list[float]) -> dict:
    a = np.array(xs)
    return {"n": len(xs), "median": round(float(np.median(a)), 4),
            "p95": round(float(np.percentile(a, 95)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4),
            "mean": round(float(a.mean()), 4)}


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "outputs" / "SYSINT"
    out.mkdir(parents=True, exist_ok=True)
    assert torch.cuda.is_available(), "no CUDA — this is the GPU reference kernel"
    device = "cuda"
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name(0)}", flush=True)

    t0 = time.perf_counter()
    from brain_tumor.inference.service import InferenceService
    svc = InferenceService(
        classifier_ckpt=root / "checkpoints" / "CLS-001" / "best.pt",
        segmenter_ckpt=root / "checkpoints" / "SEG-001" / "best.pt",
        device=device)
    torch.cuda.synchronize()
    cold_load = time.perf_counter() - t0
    print(f"cold load (import+weights): {cold_load:.2f}s", flush=True)

    images = [_synthetic(s) for s in SEEDS]
    # Warmup (discarded): kernels, allocators, first-kernel overhead.
    for img in images:
        svc.analyze(img)
        svc.consistency(img)
    torch.cuda.synchronize()

    lat: dict[str, list[float]] = {"classify": [], "segment": [],
                                   "consistency_k8": [], "analyze": []}
    for r in range(REPEATS):
        for img in images:
            for name, fn in (("classify", lambda: svc.classify(img)),
                             ("segment", lambda: svc.segment(img)),
                             ("consistency_k8", lambda: svc.consistency(img)),
                             ("analyze", lambda: svc.analyze(img))):
                t = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                lat[name].append(time.perf_counter() - t)
        if (r + 1) % 5 == 0:
            print(f"repeat {r + 1}/{REPEATS}", flush=True)

    # Peak memory over full analyze passes (fresh peak window).
    torch.cuda.reset_peak_memory_stats()
    for img in images:
        svc.analyze(img)
        svc.consistency(img)
    torch.cuda.synchronize()
    mem = {"peak_allocated_MB": round(torch.cuda.max_memory_allocated() / 2**20, 1),
           "peak_reserved_MB": round(torch.cuda.max_memory_reserved() / 2**20, 1)}

    # Determinism probe (reported, not asserted).
    a1 = svc.analyze(images[0]).model_dump()
    a2 = svc.analyze(images[0]).model_dump()
    deterministic = (a1 == a2)

    payload = {
        "precision": "FP32 reference", "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "inputs": "synthetic 512px, seeds [101,202,303]",
        "repeats_per_image": REPEATS, "cold_load_s": round(cold_load, 3),
        "latency_s": {k: _stats(v) for k, v in lat.items()},
        "peak_gpu_memory": mem,
        "gpu_deterministic_repeat": bool(deterministic),
        "floors": "FP16/ONNX deltas vs this reference: Macro-F1/Dice >= -0.01; "
                  "quantization >= -0.02 (accuracy), plus latency improvement",
    }
    (out / "gpu_latency_fp32.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps({"cold_load_s": payload["cold_load_s"],
                      "median": {k: v["median"] for k, v in payload["latency_s"].items()},
                      "p95": {k: v["p95"] for k, v in payload["latency_s"].items()},
                      "mem": mem, "deterministic": deterministic}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
