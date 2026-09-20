# System Requirements — ENG-001 Baseline (frozen)

No new requirements may be introduced under ENG-001. Changes require a new
workstream. Locked-test, model, calibration, and contract items below are
frozen; violating any is a requirements inconsistency, not an optimization.

## 1. Functional requirements

- FR-CLS-1: Classify an MRI slice into exactly one of glioma, meningioma,
  pituitary, notumor, with probabilities summing to 1.0.
- FR-CLS-2: Report calibrated confidence = max probability; state confident
  iff confidence ≥ 0.95 AND top-2 margin ≥ 0.05 (frozen T=0.5116 applied).
- FR-SEG-1: Produce a binary tumor mask at threshold 0.5 with no morphology.
- FR-LOC-1: Report bbox xyxy ints in original input pixels, centroid of the
  largest component, area = sum of qualifying components (≥10 px); empty →
  null/None + area 0.
- FR-CON-1: Report K=8, σ=0.05 perturbation agreement; flag iff < 1.0.
  Descriptive only; never alters predictions or system state.
- FR-REL-1: Report stable/review summary with basis list; clinical_meaning
  always false. Descriptive only.
- FR-QLT-1: Assess uploads against codes Q01–Q09; reject with reasons.
  Descriptive; never blocks inference by itself.
- FR-EXP-1: Provide Grad-CAM overlay, mask overlay, and text report with
  non-clinical disclaimer; outputs bit-identical to non-explain path.
- FR-API-1: Endpoints /health, /classify, /segment, /localize, /quality,
  /consistency, /analyze, /reliability, /explain with 415/413/422/500
  envelopes and no stack-trace leakage.
- FR-UI-1: Streamlit direct service calls (no HTTP), cached init, explicit
  field rendering with empty/error branches.
- FR-SYS-1: Derive system_state by frozen precedence (degraded > uncertain >
  tumor_unlocalized > tumor_localized > healthy) with contract invariants.

## 2. Non-functional requirements

- NFR-REPRO-1: Every release regenerable hashes via build_manifest.py.
- NFR-DET-1: Identical input → identical output (CPU and GPU verified).
- NFR-CPU-1: Full CPU fallback; no CUDA-only code paths in inference.
- NFR-GPU-1: FP16/autocast CUDA representation (floors Δ 0.0 / −0.000029).
- NFR-PERF-1: T4 FP16 medians — classify ≤ 0.02s, segment ≤ 0.03s,
  analyze_core ≤ 0.05s; memory ≤ 1GB. (K8 consistency ~0.32s accepted as
  the known dominant cost.)
- NFR-PORT-1: Clean clone reconstructs; dataset-absent tests skip with reason.
- NFR-OFF-1: Offline-first; no network calls in the inference path.
- NFR-ERR-1: All failure modes produce typed errors, never exceptions
  to the caller.
