# V&V-001 Baseline & Requirements Traceability (ADVANCED-R1 = 92eb2f0)

Verification workspace: pristine clone of tag ADVANCED-R1 (commit 92eb2f0, 240 files) with LFS OIDs verified against the
release manifest (0fa58033 / ce29df5e / 36091d63) without re-pulling blobs
(bandwidth-quota rationale, recorded). Weight bytes verified by OID + prior
hash checks; inference runs against local hash-verified copies. No locked
test contact in this workstream. No new requirements introduced.

## Traceability matrix

| # | Frozen requirement | Implementation | Verification (executable unless noted) |
| --- | --- | --- | --- |
| R1 | 4-class classification, calibrated (T=0.5116, τ1=0.95, τ2=0.05) | service.classify | unit/integration tests; locked N=1000 (evidence file) |
| R2 | Segmentation + localization (thr 0.5, original-px bbox/centroid/area) | service.segment/localize + extract | tests; locked N=860 |
| R3 | Uncertainty states via frozen taus | classify state branch | PBA-004; tests |
| R4 | UNC-001 K=8 consistency, descriptive | service.consistency | tests; UNC-001 val + locked evidence |
| R5 | Input-quality gate Q01–Q09 | service.quality + gate.py | tests; QUALITY validation (0/1000, 8/9) |
| R6 | Observer-only explanation | service.explain + gradcam/report | tests; EXPL-001 evidence |
| R7 | Descriptive reliability (stable/review) | service.reliability + engine | tests; REL-001 evidence |
| R8 | API endpoints + error envelopes | app/api/main.py | integration tests (415/413/422/500) |
| R9 | Streamlit direct-call UI | app/streamlit/app.py | payload/branch tests + live checks |
| R10 | Degraded paths (missing weights) | service fallbacks | degraded unit/integration tests |
| R11 | Determinism (same input → same output) | eval-mode, fixed seeds | determinism tests (service + GPU probe) |
| R12 | FP16 deployment representation | device-gated autocast | T4 floors (Δ 0.0 / −0.000029) |
| R13 | Reproducibility (hashes/manifest) | build_manifest.py | manifest regen + clean-clone validation |
| R14 | CPU fallback | no-CUDA path | local CPU CI (device=cpu suite) |

## Documentary-only verification (flagged, not hidden)

- Locked-test metric values themselves (recomputation would be new test
  contact; values verified by artifact cross-check in the viva audit).
- GPU-side behavior beyond recorded T4 evidence (no local GPU; T4 rerun
  only if a gate fails on CPU-attributable grounds).
- Examiner-facing prose (viva pack) — reviewed by audit, not executable.

## Gate sequence

1. Fresh-environment verification (pristine clone). 2. Inference-contract
verification. 3. Cross-interface equivalence. 4. Deployment-path
verification. 5. Evidence integrity. Verdict: PASS / CONDITIONAL / FAIL.
Defects: record first, fix only via new corrective branch/release.
