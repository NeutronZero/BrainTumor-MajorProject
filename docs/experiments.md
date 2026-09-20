# Experiment log (Question/Hypothesis/Baseline/Change/Dataset/Split/Seed/Config/Metrics/Result/Interpretation/Decision)

## CLS-001 — ConvNeXt-Tiny classification baseline
- Question: competitive Macro-F1 on OUR 4000/1000 split under minimal preprocessing?
- Hypothesis: prior descriptive rank-1 transfers as strong baseline (>=0.95 val Macro-F1).
- Baseline: none (first new-project run; prior numbers are evidence, not baseline).
- Change: n/a (baseline).
- Dataset: BRISC 2025 (manifest `1c9636fb…`); Split: project train 4000/val 1000
  (`7b98dd79…`/`d4bda571…`), seed 42; test LOCKED, never evaluated.
- Config: configs/experiment/CLS-001.yaml (CE ls=0.1, AdamW 3e-4 cosine,
  AMP, clip 1.0, early-stop patience 8 on val_macro_f1, ImageNet-pretrain transfer).
- Metrics (T4 full run, kernel bt-major-cls-001-convnext-tiny v8, torch
  2.10.0+cu128, 2026-09-15): val Macro-F1 0.9950 (best epoch 14/30, early
  stopped); T=0.5116 (val NLL); ECE post 0.0021; tau1=0.95, tau2=0.05
  (criterion recorded in outputs/CLS-001/calibration_frozen.json — LOCKED).
- Checkpoint: checkpoints/CLS-001/best.pt
  (SHA256 0fa58033f7460a88a68c3d7bc6ee64d91d664cf715bb6a54e9c5faf6f9a801d7,
  epoch 14, all params finite, optimizer+RNG states present).
  last.pt not retrieved (319 MB over flaky link; non-production, still on kernel).
- Smoke (CPU, random init, 1 epoch, 64/32 stratified): pipeline OK
  (outputs/CLS-001-smoke/); val F1 0.1 as expected untrained.
- Result: hypothesis CONFIRMED (>=0.95; 0.9950 ≈ prior descriptive 0.9967, no
  numeric comparison claimed).
- LOCKED TEST (PI-authorized 2026-09-16, outputs/test_evaluation_7b860dca72ea.json,
  one-way gate, no feedback): N=1000 acc 0.9950, macro-F1 0.9952, ROC-AUC OvR
  0.999936 (addendum), ECE 0.0027, uncertain 4 (0.004); per-class F1
  0.9941/0.9935/0.9967/0.9964; confusion [[253,1,0,0],[1,304,0,1],[1,1,298,0],
  [0,0,0,140]]. N=993 sensitivity: acc 0.9950/F1 0.9951/AUC 0.999936
  (deltas 3.5e-05/2.5e-05 — conclusions independent of the 7). All 7
  contaminated cases correct+certain, listed separately in the evidence file.
- Interpretation/Decision: CLOSED. No post-test tuning permitted.

## SEG-001 — vanilla U-Net segmentation baseline
- Question: competitive val mean Dice on OUR seg-train/seg-val under minimal preprocessing?
- Hypothesis: binary U-Net forms a viable engineering baseline.
- Dataset: 4793 Gate-0-verified pairs; seg-val = seg-train images whose stems
  map to project val (contamination-free by construction); test 860 LOCKED.
- Config: configs/experiment/SEG-001.yaml (Dice+BCE summed, AdamW 3e-4,
  early-stop patience 8 on val_dice, mask convention `>127->1`).
- Smoke (CPU, 1 epoch, 32 pairs): pipeline OK (outputs/SEG-001-smoke/).
- Metrics (T4 full run, kernel bt-major-seg-001-vanilla-u-net v1, torch
  2.10.0+cu128, 2026-09-16): val mean Dice 0.8731, 30/30 epochs (no early
  stop); norm from OUR seg-train (mean 0.10168, std 0.14064); seg-train pairs
  3148 / seg-val 785, 0 missing, split rule verified in-kernel.
- Checkpoint: checkpoints/SEG-001/best.pt (state-only repack of kernel
  best.pt, exact fp32, SHA256 ce29df5e…2b80 verified against kernel sidecar;
  epoch 27, val 0.8731, 31,036,481 params, all finite, forward-verified
  locally into our UNet). Full checkpoint (optimizer/RNG) remains on the
  kernel; production registry needs state only.
- Result: viable engineering baseline (0.8731 val Dice).
- Validation analysis (no test; outputs/SEG-001-analysis/seg_val_analysis.json):
  recomputed macro mean Dice 0.8518 (median 0.9298, IQR 0.845–0.963, p10 0.651;
  training-reported 0.8731 is micro-style — definition difference, not drift);
  train subset 0.787 (glioma-heavy first-200 order effect, documented);
  per-class means glioma 0.705 / meningioma 0.937 / pituitary 0.890 (glioma
  infiltrative boundaries = known hard cases; worst-10 recorded);
  localization invariants hold on all 785 (14 empty->null/0, 771 boxes in-frame,
  60 multi-component warnings); GT never empty. NO pipeline change required.
- LOCKED TEST (same evidence file): seg-test N=860 mean Dice 0.8617
  (median 0.9385, p10 0.656, IoU 0.7934; 10 empty preds, 81 multi); N=853
  sensitivity Dice 0.8610 (delta 0.0007). Disagreement confident-tumor +
  empty-seg: 10/856 = 1.17% (< 5% — reported, not repaired; per-class
  glioma 7 / meningioma 1 / pituitary 2). System states: localized 846,
  unlocalized 10, uncertain 4, healthy 140.
- Interpretation/Decision: CLOSED. No post-test tuning permitted.

## Post-Baseline Decision Gate (PBA-001..005 CLOSED)
- Findings: noise σ=0.05 is the dominant stressor (CLS acc −0.019 with 20/24
  errors confident-but-wrong, ECE→0.0215; SEG Dice −0.169, empty 10→163);
  JPEG q=50 and downscale 0.75× negligible; glioma weakest seg class throughout.
- Decision: open narrow campaign. H-ROBUST-001 (robustness) + H-UNCERT-001
  (uncertainty) as SEPARATE experiments. CLS-001/SEG-001 frozen; no baseline
  modification, no test-driven tuning, no aggregate score, no OOD claims.
- UNC-001: QUEUED (validation-only mechanism research; not started).

## UNC-001 — perturbation-consistency uncertainty (AUTHORIZED, validation-only)
- Hypothesis H-UNCERT-001: K=8 fixed σ=0.05 probes (exact PBA-003 impl;
  seeds 7003+idx*16+k; eval-noise 9000+idx) + pre-registered FLAG
  (agreement<1.0). Frozen CLS-001/T/taus; design in
  outputs/UNC-001/design_freeze.json (written pre-execution). NO test contact.
- Result (val N=1000; outputs/UNC-001/unc001.json): eval-noise errors 24;
  flagged 18/24 (0.75) vs confidence-only 6/24 (0.25); CBW 17/21 (0.81).
  Burden: 28/1000 flagged (0.028); false alerts 27/995 clean-correct (0.027).
  Agreement 8/8 on 972 cases; 5 flags from single-probe flips. By class:
  meningioma 14 errors/0.86 detected but 0.061 false-alert; pituitary 0 noise
  errors; glioma 8 errors/0.625. Clean confusion [[226,4,0,0],[0,264,1,0],
  [0,0,290,0],[0,0,0,215]] (5 clean errors, consistent with CLS-001 val).
- Interpretation: FOR PI JUDGMENT against the pre-registered criterion. No
  score constructed. Locked evaluation NOT authorized.
- LOCKED TEST (PI-authorized, Kaggle CPU kernel v3; frozen procedure, no
  re-cut; outputs/UNC-001/unc001_locked.json): N=1000, eval-noise errors 27;
  flagged 25/27 (0.926) vs confidence-only 5/27 (0.185); CBW 24/25 (0.96).
  Burden: 36/1000 flagged (0.036); false alerts 33/995 clean-correct (0.033).
  Agreement 8/8 on 964 cases. By class: glioma 13 errors/0.92 detected
  (false-alert 0.067); meningioma 14/0.93 (0.049); pituitary 0 errors, 0.0
  false-alert; notumor 0 errors. Validation criterion passed on locked
  population with detection stronger than validation (0.75->0.93) at
  comparable burden (0.027->0.033).
- Status: UNC-001 CLOSED. Failure-detection layer documented alongside the
  frozen CLS-001 baseline; not a replacement classifier.

## ROB-001 — train-only Gaussian-noise U-Net (AUTHORIZED, not run)
- Question: does train-only noise (σ=0.05) improve noise-robustness without
  harming clean val Dice?
- Change vs SEG-001 (single factor): augmentation.gauss_noise_sigma=0.05
  (photometric, mask/geometry identical per plan §30; default 0.0 elsewhere).
- Config: configs/experiment/ROB-001.yaml; script: scripts/train/train_rob001.py
  (writes checkpoints/ROB-001/, outputs/ROB-001/; SEG-001 files untouched).
- Kernel: kaggle/rob001/ (T4 GPU). Selection: seg-val Dice. Test LOCKED.
- Status: scaffolded + unit-tested locally; training NOT started (runs on Kaggle).
- T4 run (kernel bt-major-rob-001-noise-u-net, ~68 min, 30/30 epochs,
  best val 0.8337 micro-style, norm identical to SEG-001, config SHA verified,
  ckpt epoch 21/30 epoch-index nuance from min_delta — plateau, not a fault):
  artifacts checkpoints/ROB-001/best.pt + outputs/ROB-001/ (last.pt on kernel).
- Validation-only 2x2 (frozen seg-val N=785, macro-style, exact PBA-003 noise
  path; outputs/ROB-001/val_2x2.json):
  A. clean: SEG 0.8518 / ROB 0.8070 (d_mean -0.0447, d_median -0.0237,
     d_p10 -0.1696; per-class glioma 0.705->0.615, meningioma 0.937->0.917,
     pituitary 0.890->0.859; multi 60->150).
  B. noise s=0.05: SEG 0.8518->0.6786 (deg -0.1731, empty 14->156, glioma
     ->0.319); ROB 0.8070->0.8135 (deg +0.0064, empty 10->13, glioma 0.628).
  Gap closed 0.1795. Robustness effect demonstrated WITH clean cost concentrated
  in the lower tail. No feedback into ROB-001; locked eval NOT authorized.
- LOCKED TEST (PI-authorized, outputs/ROB-001/locked_eval.json + freeze.json;
  seg-only, frozen CLS-001 records reused, no new CLS inference):
  N=860 mean Dice 0.8207 (median 0.9197, p10 0.5435, IoU 0.7435; empty 9,
  multi 163); per-class glioma 0.6625 / meningioma 0.9247 / pituitary 0.8487.
  N=853 sensitivity Dice 0.8197 (delta vs primary -0.0010; vs SEG-001 -0.0412).
  Deltas vs SEG-001 locked: mean -0.0409 / median -0.0188 / p10 -0.1127 /
  IoU -0.0499 — clean cost reproduces the validation estimate (-0.045).
  Disagreement 9/856 = 1.05% (< 5%; all glioma). System states: localized 847,
  unlocalized 9, uncertain 4, healthy 140.
- Interpretation: H-ROBUST-001 supported (validation); locked test confirms the
  pre-registered trade-off shape. Preferability is a project decision, not
  established by these numbers alone. CLOSED to tuning.
- PI decision: ROB-001 CLOSED. SEG-001 remains the clean-distribution baseline;
  ROB-001 the documented robustness-oriented variant; neither substitutes for
  the other. No noise-perturbed locked-test probe authorized (robustness was
  pre-registered validation-only). Next: Post-ROB Decision Gate.

## Final system selection (PI decision — model development finished)
- Default path: CLS-001 (acc 0.9950, F1 0.9952, ECE 0.0027; T=0.5116,
  tau1=0.95, tau2=0.05) + UNC-001 consistency layer (inference-time warning
  only; CLS-001 predictions unchanged) + SEG-001 (test Dice 0.8617) + frozen
  localization. ROB-001 kept as documented non-default robustness variant
  (test Dice 0.8207; not a replacement).
- Claim boundary: 99.50% is a measured BRISC locked-test result under the
  frozen regime, not clinical diagnostic accuracy. Known limitations stand:
  cross-split contamination (disclosed, sensitivity-verified), no
  subject-level independence, 140 classification-only healthy cases without
  seg GT, noise sensitivity, research prototype (not a clinical device).
- Next phase: System Integration & Engineering Validation (no new models).

## System integration (in progress — no new models)
- Wired frozen CLS-001 + SEG-001 + T/taus + seg norm into InferenceService;
  endpoints /health /classify /segment /localize /consistency /analyze
  (analyze carries a descriptive `consistency` payload; no state gating —
  gating needs a contract amendment, deferred as a PI decision).
- CPU CI: 23 passed (tests/unit + tests/integration, synthetic images only,
  zero locked-test contact). Legacy dummy-era contract test updated.
- Laptop-CPU latency (512px synthetic): classify 0.22s, segment ~2.0s,
  consistency(K=8) ~2.1s, analyze ~1.4s. GPU/optimization ladder pending.
- Streamlit check PASSED via live user upload: direct service import (no HTTP),
  cache_resource init, bytes reach service, all fields render (class,
  confidence, states, probabilities, consistency, bbox/centroid/area);
  bbox/centroid coherence verified on the live output. Payload + branch tests
  in tests/integration/test_streamlit.py (synthetic only).
- GPU FP32 reference (T4 kernel v2; outputs/SYSINT/gpu_latency_fp32.json):
  cold load 9.69s; medians classify 0.0111s / segment 0.0242s /
  consistency-K8 0.3397s / analyze 0.0349s; p95s 0.0137/0.0281/0.3688/0.0393;
  peak allocated 370MB / reserved 576MB; GPU repeat-deterministic. v1 exposed
  a real service bug (models never moved to device) — fixed, CPU suite green.
  Reference frozen for the FP16/ONNX/quantization ladder floors.
- FP16 rung PASSED (T4 kernel v1; outputs/SYSINT/fp16_check.json; autocast,
  weights untouched): val floors macro-F1 Δ 0.0 / Dice Δ -0.000029 (both PASS
  vs -0.01). Latency FP16 vs FP32 medians: classify 0.0125/0.0111, segment
  0.0178/0.0242, K8 0.318/0.340, analyze 0.0301/0.0349; memory unchanged
  (weights stay FP32); FP16 repeat-deterministic. FP16 is the candidate
  deployment representation; accuracy and latency kept as separate quantities.
- ONNX rung (T4 kernels v1-v4; outputs/SYSINT/onnx_check.json): correctness
  PASSED — ORT-CUDA val deltas F1 0.0 / Dice 0.0, localization empty+bbox
  100/100, deterministic. Deployment requirement NOT met: ORT-CUDA medians
  classify 0.0102 / segment 0.0245 / K8 0.305 / analyze 0.0353 — parity with
  torch FP32 (0.0111/0.0242/0.340/0.0349), no improvement; peak 917MB
  (ORT arena pre-alloc vs torch 370/576). v3 ran silently on CPU EP
  (invalid deployment numbers, now superseded); v4 pins ORT 1.21.x + asserts
  CUDA engagement. Decision: retain FP16 representation; ONNX documented as
  accurate-but-no-faster. K8 dominates both runtimes; its batching stays a
  separate rung.
- K=8 batching rung REJECTED (T4 BENCH-004; outputs/SYSINT/batchk8_check.json):
  equivalence exact (0/1000 agreement + flag mismatches; untouched F1/Dice
  reproduce FP32; deterministic) but batched 0.258s is SLOWER than sequential
  0.159s same-run (speedup 0.61x) — fails the benefit condition, sequential
  retained and reverted in service.py (CI 27 green). Corrected accounting:
  analyze_core 0.0282, analyze_full 0.2861, consistency cost 0.258 (batched)
  / 0.131 implied sequential. Note: sequential K8 varied 0.34->0.16 across
  kernels (shared-host variance) — within-run interleaving is what counts.
- Optimization ladder CLOSED (PI decision): FP16/autocast = final deployment
  representation (floors PASS, deterministic); ONNX accurate-but-no-faster;
  batching rejected; quantization NOT opened (no bottleneck justifies the
  -0.02 budget: core pipeline ~30ms, dominant cost is the sequential K8 layer
  batching cannot fix). FP16 frozen into service via device-gated autocast
  (no-op on CPU). Deployment validation: CPU CI 28 green (incl. degraded
  paths, error envelopes, determinism); final T4 numbers = FP16 rung medians
  (classify 0.0125 / segment 0.0178 / K8 0.318 / analyze 0.0301; peak
  370/578MB).

## Workstream 2 — EXPL-001 explainability (advanced branch; SB-1 untouched)
- Grad-CAM observer on CLS-001 (target `features.7.2.block.0`, hooks only,
  no retraining; FP32 explanation forwards on all devices). Contribution
  visualization only — no attention or clinical-meaning claim.
- Unified object: inference outputs (provably identical) + consistency +
  quality + heatmap/seg overlays (base64 PNG) + JSON/text report with
  disclaimer. `/explain` endpoint; Streamlit Explain button.
- Evidence `outputs/EXPL-001/expl001.json`: observer equivalence true,
  deterministic, disclaimer present. CPU CI 37 green (synthetic only).
- Live-check diagnostic (advanced branch): `focus_in_bbox` 0.00 verified as a
  CORRECT near-zero measurement, not a bug — same coordinate system
  (bbox original-px, CAM resized pre-statistic), normalized-float (not JET)
  representation, dedicated `cam_mass_in_bbox` helper with 1.0/0.0/None unit
  coverage; display precision raised to 4 decimals to rule out rounding
  ambiguity. Interpretation: Grad-CAM mass fell outside the segmented bbox
  for that image — consistent with the no-attention-claim boundary. No
  model/Grad-CAM math change. CPU CI 40 green.

## Workstream 1 — input-quality gate (SB-1 fallback intact)
- Deterministic gate (`src/brain_tumor/quality/gate.py`, codes Q01-Q09,
  fixed constants, no model): file/format/dimension/pixel/finite/
  blank/intensity/content/size/aspect checks. SB-1 inference untouched.
- Validation (`outputs/QUALITY/gate_validation.json`; val-only reporting, no
  test contact): 0/1000 false rejects on clean val; 8/9 corruption battery
  detected (pure noise passes pixel stats — documented boundary; semantic
  anomaly detection is future OOD work, not this rung).
- Wired as descriptive `service.quality()` + `/quality` endpoint; does not
  alter outputs or system_state. CPU CI 34 green.

## Workstream 3 — REL-001 reliability engine (advanced branch; SB-1 untouched)
- Descriptive fusion only (`src/brain_tumor/reliability/engine.py`, pure
  function): confidence + consistency + quality + seg/localization info.
  Fixed rule — stable iff confident + agreement 1.0 + quality accept +
  (notumor OR nonempty+localized), else review with basis list. No score,
  no retraining, no contract/threshold/state changes, clinical_meaning false.
- Surfaced as `service.reliability()` + `/reliability` + /analyze payload +
  Streamlit lines. Evidence `outputs/REL-001/rel001.json` (rule, hashes,
  synthetic combination matrix). CPU CI 48 green (synthetic only).
- Quarantine (2026-09-20): an independent audit session ran inference on
  locked-test image `brisc2025_test_00001_gl_ax_t1.jpg` (direct script +
  Streamlit AppTest). Unplanned smoke contact, NOT evaluation evidence:
  enters no output file, calibration, selection, or comparison; no decision
  flows from it. Full suite per that audit: 49 passed (incl.
  tests/data/test_gate_fail_open.py).

## Packaging hardening (forensic-audit response; no model/evidence changes)
- Fixed `.gitignore` (`data/` → `/data/`); committed `src/brain_tumor/data/`,
  `configs/data/`, `tests/data/` (49th test restored).
- Removed all machine-specific paths (3 scripts + pba003 docstring +
  brisc2025 `copied_from` provenance note); UTF-8 manifest; regenerated
  manifest; dataset tests skip cleanly without BRISC bytes; cam test moved
  to `tests/unit/test_explain_units.py`.
- Clean-clone validation from GitHub PASSES: 47 passed + 2 skipped (no
  BRISC bytes), synthetic inference + API healthy, manifest regenerates
  with checkpoint hashes matching.
- Releases: `ADVANCED-R1` (advanced @92eb2f0); `SB-1.1` (branch from SB-1,
  packaging-only delta, suite 35 green) — original tags immutable.
- Process incident (recorded, not hidden): three near-identical old/new-string
  edit operations corrupted file structure (joined lines); each detected via
  compile/tests/header checks and repaired immediately. Standing rule: never
  issue a near-identical replacement without verifying the exact intended
  edit first; verify headers after every log edit.

## CORRECTION-001 — forensic-audit findings resolved (branch correction-001)
- C1 LF policy (`* text=auto eol=lf` in .gitattributes); manifest regenerated
  UTF-8 with corrected CLS hash below.
- C2 reproducibility_check.py rebuilt: deterministic synthetic input + real
  assertions (class/confidence/state repeat, probs sum); exits 0.
- C3 checkpoint hygiene (Option A): CLS-001 best.pt replaced by key-removal
  derivative (451e4fc4…, 111MB vs 334MB; 182 tensors identical,
  bit-identical inference on 3-case battery, 27.8M params). Original retained
  in git history with lineage record in the file; locked eval stays
  attributed to the original. Service loads ["state"] from either.
- C4 README plan reference made repo-relative.
- Auditor wording corrected in our record: "no data leakage" is NOT adopted;
  standing language is "no test-driven tuning found; disclosed cross-split
  contamination; never leakage-free/patient-independent".
- Full CI 49 green; manifest complete with zero MISSING.

## ENG-002 — final system productization (branch eng-002; no model changes)
- UI: input image displayed; quality section + model-vs-observer labeling in
  Analyze/Explain views. API: quality attached to /analyze payload.
- Demo suite (`scripts/demo/demo_matrix.py`, `outputs/DEMO/demo_matrix.json`):
  7 deterministic synthetic contract cases (healthy/tumor/low-quality/
  seg-empty/seg-localized/uncertain/degraded), all holding; drift means the
  fixture is replaced, never the model.
- Docs: `docs/api/endpoints.md`, `docs/user_guide/usage.md`,
  `docs/deployment/profiles.md`. Full CI 49 green.

## Full-run commands (Kaggle T4, network only for pretrained weights fetch)
- `python scripts/train/train_classifier.py`  (≈1–3 GPU-h budget, §55)
- `python scripts/train/train_segmenter.py`   (≈3–6 GPU-h budget, §55)
- Calibration/taus fit on validation only inside each script; test evaluated
  exactly once afterwards via scripts/evaluate/run_test.py (still locked).
