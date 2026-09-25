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
- Batch: `/analyze_batch` bounded (REL-002) — max 8 files, 32MB aggregate
  bytes, 64MP aggregate decoded pixels; rejections are 413 before full parse.

## Container (REL-002)

- Build: `docker build -t braintumor-api:<TAG> .` — image ships source +
  locked deps only (`requirements.lock`); `checkpoints/` and `outputs/` are
  NOT baked in. Mount them read-only at runtime:
  `docker run --rm -p 8000:8000 -v ./checkpoints:/app/checkpoints:ro -v ./outputs:/app/outputs:ro braintumor-api:<TAG>`
  Artifact identity remains governed by the release manifest hashes, not the
  image build cache.
- **Rollback = redeploy the previous immutable image tag.** Tags are
  release-scoped (e.g. `braintumor-api:SB-1`, `braintumor-api:REL-002`); never
  mutate a published tag — build a new one. No orchestrator is in scope; this
  is the whole rollback procedure.
- Deployment boundary: loopback / internal research demo (see ethics.md and
  the API disclaimers). Auth, rate limiting, CORS, and reverse-proxy TLS are
  OUT of scope for this profile and become release requirements only if the
  target changes to LAN/public exposure — an explicit future decision (audit
  finding #4, deliberately deferred, not silently accepted).
