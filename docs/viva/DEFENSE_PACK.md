# Viva Defense Pack — frozen against ADVANCED-FOUNDATION (454f207)

Every answer below ties to a frozen implementation or evidence artifact.
Research prototype — not clinical. All numbers are measured BRISC-regime
engineering results, never clinical-accuracy claims.

## 1. Project overview

End-to-end brain-tumor MRI analysis platform: classification (CLS-001),
segmentation + localization (SEG-001), failure detection (UNC-001),
input-quality gate (Q01–Q09), explanation views (EXPL-001), descriptive
reliability reporting (REL-001), served over FastAPI + Streamlit from one
shared InferenceService. Documented robustness variant ROB-001 (non-default).
Deployment: FP16/autocast, CPU fallback verified.

## 2. Problem statement and requirements

Requirements: 4-class slice classification with calibrated uncertainty;
tumor segmentation with bbox/centroid/area; explicit failure signaling;
deployability on GPU and CPU. Deliberately scoped AWAY from diagnosis:
the system reports measurements and states, never medical conclusions.

## 3. Why PyTorch

Ecosystem standard for the referenced architectures; deterministic eval
paths; TorchScript/ONNX export routes for the deployment ladder; AMP for
training stability. Pinned: torch 2.10.0+cu128 (kernels), 2.13.0+cpu (local).

## 4. Why ConvNeXt-Tiny

Prior-project controlled comparison (C0–C5 × seeds) ranked it first
descriptively (val 0.9967) — used as evidence for baseline prioritization,
NOT as a superiority claim (C4 vs C3 statistically indistinguishable,
McNemar p≈0.7–1.0; see architecture_evidence.md). New-project run confirmed
it as a strong baseline (val F1 0.9950). Rationale documented before training.

## 5. Why U-Net

Vanilla 2D U-Net per plan S43 as the minimal viable segmentation baseline:
independent resolution/norm/augmentation/training decisions, binary mask
convention from Gate 0. It established 0.8617 test Dice — sufficient for an
engineering baseline without architectural novelty claims.

## 6. Dataset provenance and Gate 0

BRISC 2025 via Kaggle `briscdataset/brisc2025` (CC BY 4.0). Gate 0 verified
hashes, class counts (train 1147/1329/1067/1457; test 254/306/140/300),
4793 masks (`>127→1`, files untouched), heterogeneous resolution, and
audits (exact/near-duplicate, split-overlap). Verdict: FAIL with
conditional pass — contamination found, protocol defined, project continued
under explicit PI authorization rather than silent data surgery.

## 7. The 7 cross-split contamination cases

Seven byte-identical image hashes span train↔test in the RELEASED data
(discovered, not introduced, by us). All 7 listed separately in the locked
evidence file; all correctly classified and certain; N=993 sensitivity
deltas 3.5e-05/2.5e-05 — conclusions independent of them. Wording locked:
"reported on N=1000 with 7 disclosed; N=993 essentially unchanged" — never
"leakage-free".

## 8. Why no patient-level independence claim

Slice-level release without complete subject-identity mapping. Stating
independence would require evidence we do not have; the limitation is
declared in the data card and limitations report instead.

## 9. Train/validation/test separation

5000-image pool → frozen 4000/1000 split (seed 42, manifest hash-pinned);
official test 1000 locked behind a one-way gate (evaluated exactly once per
experiment, never fed back). Calibration, thresholds, and model selection
are validation-only. Seg-val (785) derives from the same project split by
stem mapping — contamination-free by construction.

## 10. Calibration

T=0.5116 (val NLL), τ1=0.95/τ2=0.05 (val criterion), frozen pre-test.
Test ECE 0.0027; independently recomputed == stored == locked (tol 1e-6).
Reliability diagram on file. Caveat preserved: 996/1000 cases sit in the top
confidence bin — low scalar ECE ≠ uniform calibration; only 4 uncertain cases.

## 11. Segmentation and localization

Threshold 0.5, no morphology; bbox xyxy ints in ORIGINAL pixels, centroid of
largest component, area = sum of components; empty→null/0 enforced by
contract validators. Invariants hold on all 785 val pairs; multi-component
surfaced as warning, never silently merged.

## 12. UNC-001

K=8 fixed σ=0.05 probes (PBA-003 implementation); flag iff agreement < 1.0.
Validation: 18/24 noise errors (vs 6/24 confidence-only), CBW 17/21, burden
2.7%. Locked: 25/27 (0.926) vs 5/27, CBW 24/25, burden 3.3%. Descriptive
layer; predictions and system_state unchanged by design.

## 13. Quality gate Q01–Q09

Deterministic checks (decodability, size, dimensions, pixels, finite,
blank, intensity, content, aspect) with fixed constants. 0/1000 false
rejects on clean val; 8/9 corruption battery detected (pure noise passes
pixel stats — documented boundary; semantic anomalies are future OOD work).

## 14. EXPL-001 and the Grad-CAM limitation

Hooks-only observer on `features.7.2.block.0`; explain=true outputs
bit-identical to explain=false; deterministic; FP32 forwards. The verified
0.00 focus-in-bbox case is our concrete evidence that Grad-CAM ≠ tumor
attention — the boundary is demonstrated, not merely asserted.

## 15. REL-001

Pure fusion (confidence + consistency + quality + seg/localization):
stable iff confident + 1.0 agreement + quality accept + (notumor OR
localized), else review with basis list. No score, no override,
clinical_meaning always false. Combination matrix + 48-test CI on file.

## 16. ROB-001

Single-factor train-noise (σ=0.05) U-Net. Eliminated noise degradation
(−0.173 → +0.006 val) at a reproducible clean cost (−0.045 val, −0.041
locked). Supported hypothesis, non-default variant, closed to tuning.

## 17. Robustness findings (PBA-003, plan §68 suite)

Noise σ=0.05 is the dominant stressor (CLS −0.019 with 20/24 CBW and ECE→
0.0215; SEG −0.169 with empty 10→163). JPEG q=50 and downscale 0.75×
negligible. Glioma weakest under every perturbation. No aggregate score
(by charter).

## 18. FP16 / ONNX / batching decisions

FP16/autocast: floors Δ 0.0 / −0.000029 → PASS, retained, deterministic.
ONNX: numerically faithful (Δ 0.0, localization 100/100) but no latency gain
at higher memory → closed, PyTorch retained. K=8 batching: exact (0/1000
mismatches) but 0.61× speedup → rejected, sequential retained. Quantization
never opened (no bottleneck justified the −0.02 budget). Accuracy and
latency always reported separately.

## 19. CPU fallback

Device-gated autocast (no-op on CPU); missing-checkpoint degraded paths
unit-covered; full CPU CI green; laptop-CPU latencies recorded as baseline.

## 20. API + Streamlit architecture

One InferenceService; FastAPI thin wrapper (/health, /classify, /segment,
/localize, /quality, /consistency, /analyze, /reliability, /explain) with
415/413/422/500 envelopes and no stack leakage; Streamlit direct-calls the
service (no HTTP), cache_resource init, explicit field rendering with
empty/error branches. Live-check passed; two historical UI bugs (healthy
path, large-image overlay) found by live testing, fixed, regressed.

## 21. Failure/degraded paths

Segmenter-missing → degraded; undecodable/oversize → envelopes; blank →
quality reject + review (inference still runs); uncertain → low_confidence
warning; empty seg + tumor → tumor_unlocalized + warning. All covered in CI.

## 22. Performance and memory

T4 FP16 medians: classify 0.0125s, segment 0.0178s, K8 0.318s, analyze
0.0301s (core 0.0282 + consistency); peak 370/578MB; cold load ~9.7s;
GPU repeat-deterministic. CPU baselines on file. K8 dominates by design.

## 23. Known limitations

Noise sensitivity; confidence-only uncertainty gaps (UNC-001 catches most,
3.3% burden); glioma lower tail; fragmentation under ROB-001; released-data
contamination; no subject independence; 140 healthy cases without seg GT;
heterogeneous resolution; no OOD claim; single-release validity.

## 24. Why an engineering prototype, not a clinical system

No clinical validation, no multi-site data, no radiologist study, no
regulatory standing, no generalization evidence. The UI header, API payloads,
and every report carry the non-clinical boundary. The project validates
engineering method, not medical utility.

## 25. Likely examiner attack questions

- "Why not a bigger/newer backbone?" → Baseline-first method; rank-1
  evidence with statistical tie disclosed; no superiority claim needed for
  an engineering baseline; ladder evaluated deployment, not architectures.
- "Why is your Dice 'only' 0.86?" → Macro mean with p10/median reported;
  glioma boundary ambiguity characterized; no SOTA claim made.
- "Did you tune on test?" → One-way gate per experiment; frozen manifests;
  sensitivity analyses; quarantine log for accidental contacts.
- "Why keep SEG-001 over ROB-001?" → Locked test reproduced the clean cost;
  preferability is a requirements decision, both documented.
- "What if inputs are garbage?" → Q01–Q09 gate + error envelopes + degraded
  paths, all tested.
- "Isn't 28→48 tests thin?" → Each test pins a contract/invariant; suite
  grew with every bug found (healthy path, overlay size, no-op edit guards).

## 26. Short explanations

- 30s: "Validated MRI analysis prototype: 99.5% classification and 0.86
  Dice on locked BRISC evaluation, with calibrated uncertainty,
  failure detection, and a tested API — documented as engineering, not
  clinical."
- 2min: add frozen methodology (validation-only selection/calibration,
  one-way test gate, sensitivity analyses), robustness trade-off, descriptive
  reliability layer, FP16 deployment numbers.
- 5min: add Gate-0 contamination handling, PBA failure analysis, UNC-001
  locked generalization, ladder decisions with floors, SB-1/ADVANCED
  fallback structure, limitations.

## 27. Evidence-to-claim matrix

| Claim | Evidence artifact |
| --- | --- |
| CLS-001 0.9950/0.9952 | outputs/test_evaluation_7b860dca72ea.json (+AUC addendum) |
| ECE 0.0027, 4 uncertain | same + outputs/PBA-004 (plot+data, recomputed match) |
| SEG-001 Dice 0.8617 | same evidence file, seg blocks |
| Disagreement 1.17% < 5% | same + PBA-005 denominator table |
| Contamination immaterial | N=993/N=853 sensitivity deltas in same file |
| Noise failure mode | outputs/PBA-003 (per-arm, control-anchored) |
| UNC-001 0.926/0.033 locked | outputs/UNC-001/unc001_locked.json (+freeze) |
| ROB-001 trade-off | outputs/ROB-001/val_2x2.json + locked_eval.json |
| FP16 floors pass | outputs/SYSINT/fp16_check.json |
| ONNX faithful/slower | outputs/SYSINT/onnx_check.json |
| Batching rejected | outputs/SYSINT/batchk8_check.json |
| Quality 0/1000, 8/9 | outputs/QUALITY/gate_validation.json |
| EXPL-001 observer-safe | outputs/EXPL-001/expl001.json |
| REL-001 descriptive | outputs/REL-001/rel001.json |
| SB-1 integrity | docs/release_manifest.md (recomputable hashes) |
| CI 48 green | tests/unit + tests/integration (synthetic only) |

## Overclaiming interrogation (examiner-grade)

**Q: "Your accuracy is 99.5%. Can you say your system is 99.5% accurate
for brain-tumor diagnosis?"**
A: No. 99.50% is the measured accuracy on the 1000-image locked BRISC test
population under our frozen preprocessing and evaluation code. It says
nothing about other hospitals, scanners, populations, or clinical diagnosis,
and our data card explicitly disclaims subject-level independence while
disclosing cross-split contamination. The correct statement is always the
regime-bound one.

**Q: "Does your Grad-CAM prove the model is looking at the tumor?"**
A: No — and we have a verified counterexample: a locked-case-style
meningioma prediction whose CAM mass inside the segmented bbox measured
effectively zero while every other output was coherent. Grad-CAM visualizes
output contribution, nothing more; the report carries that boundary in text.

**Q: "Does 0.926 UNC-001 detection mean shifts are handled?"**
A: Only the tested shift: σ=0.05 Gaussian noise. Other corruptions,
scanners, and true OOD remain unvalidated — the limitations report says so
explicitly, and the burden (3.3% false alerts) travels with every citation
of the detection rate.

**Q: "Is ROB-001 better than SEG-001?"**
A: Unaskable without "at what?": +0.135 under noise, −0.041 clean, worse
p10 and glioma. Neither replaces the other; requirements decide.
