# Deployment Profiles — ENG-002 (frozen decisions)

## Representations

- Production: FP16/autocast, CUDA-gated (CPU executes FP32-identical path).
  Floors verified: Macro-F1 Δ 0.0, Dice Δ −0.000029, deterministic.
- Not adopted: ONNX (faithful, no latency gain, more memory), K=8 batching
  (exact, slower), quantization (never opened — no bottleneck justified it).

## Profiles

- **CPU demo**: any machine with the repo + deps; medians ~0.22s classify /
  ~2.0s segment / ~2.1s K8 / ~1.4s analyze (512px, laptop reference).
- **GPU (T4)**: FP16 medians 0.0125 / 0.0178 / 0.318 / 0.0301s; peak
  370MB allocated / 578MB reserved; cold load ~9.7s; repeat-deterministic.
- **Kaggle**: bench/rob/seg/unc kernels document the exact launch protocol.

## Operational checks

- Startup: both checkpoints load (paths in release manifest).
- Health: `/health` reports model + calibration status; `degraded` when the
  segmenter is absent.
- Integrity: rerun `scripts/release/build_manifest.py` — zero MISSING.
- Limits: 10MB uploads; typed errors; offline-capable inference path
  (OFF-001 PASS with documented loopback exemption).
