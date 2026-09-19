# Model Card — Submission Baseline SB-1 (frozen)

Engineering model documentation. Reconstructed strictly from frozen evidence
artifacts; no test-set contact was used for documentation. Research prototype
— not clinical. Measured BRISC results below are regime-specific engineering
validation, not clinical diagnostic accuracy claims.

## System composition (final selection)

| Role | Component | Disposition |
| --- | --- | --- |
| Classifier (default) | CLS-001 ConvNeXt-Tiny | Primary clean-distribution path |
| Segmentation (default) | SEG-001 vanilla U-Net (2D, base 64/128/256/512, bottleneck 1024) | Primary clean-distribution path |
| Failure detection | UNC-001 perturbation-consistency (K=8, σ=0.05) | Descriptive layer; does not alter predictions or system_state |
| Robustness variant | ROB-001 noise-trained U-Net | Documented alternative; NOT default (clean trade-off) |
| Production representation | FP16/autocast, CUDA-gated (CPU: FP32-identical no-op) | Frozen deployment choice |

## CLS-001 — classification

- Architecture: ConvNeXt-Tiny, ImageNet-pretrained init (explicit transfer
  decision), 4-output head; seed 42. Config `configs/experiment/CLS-001.yaml`
  (`0565364a…`); preprocessing `configs/preprocessing/cls_baseline.yaml`
  (`1d0986ec…`): grayscale→3ch, resize 224, ImageNet normalize; train-only
  hflip/rotation/affine; loss CE label-smoothing 0.1; AdamW 3e-4 cosine,
  AMP, clip 1.0, early-stop patience 8 on val Macro-F1.
- Checkpoint `checkpoints/CLS-001/best.pt` (`0fa58033…`), best epoch 14/30.
- Validation (project val N=1000): Macro-F1 0.9950. Frozen calibration (val
  NLL/val criterion, test never touched): T=0.5116, τ1=0.95, τ2=0.05
  (`outputs/CLS-001/calibration_frozen.json`, `2d4bd736…`).
- Locked test (N=1000, `outputs/test_evaluation_7b860dca72ea.json`,
  `9dbaefdd…`): accuracy **0.9950**, macro-F1 **0.9952**, ROC-AUC OvR
  0.999936 (addendum), ECE **0.0027**, uncertain 4 (0.4%); per-class F1
  glioma 0.9941 / meningioma 0.9935 / pituitary 0.9967 / notumor 0.9964;
  5 errors. N=993 sensitivity: acc/F1 deltas 3.5e-05/2.5e-05 (7 disclosed
  cross-split cases all correct+certain; conclusions independent of them).

## SEG-001 — segmentation

- Architecture: vanilla U-Net (31,036,481 params); Dice+BCE summed; AdamW
  3e-4; patience 8 on val Dice. Mask convention `>127→1` (Gate-0). Config
  `configs/experiment/SEG-001.yaml` (`69b3ead6…`); norm from seg-train
  (mean 0.10168, std 0.14064).
- Checkpoint `checkpoints/SEG-001/best.pt` (`ce29df5e…`), epoch 27/30.
- Validation: training-reported 0.8731 is a micro-style aggregate; recomputed
  macro mean Dice 0.8518 (median 0.9298, p10 0.651); per-class glioma 0.705 /
  meningioma 0.937 / pituitary 0.890 (glioma infiltrative boundaries = known
  hard cases). Localization invariants hold on all 785 val pairs.
- Locked test (N=860, same evidence file): mean Dice **0.8617** (median
  0.9385, p10 0.656, IoU 0.7934; 10 empty, 81 multi). N=853: 0.8610
  (delta 0.0007). Disagreement confident-tumor+empty-seg 10/856 = 1.17%
  (< 5% threshold; glioma 7 / meningioma 1 / pituitary 2).

## UNC-001 — failure-detection layer

- Frozen CLS-001 + K=8 fixed σ=0.05 probes (PBA-003 implementation), flag iff
  probe agreement < 1.0. No retraining, no refit, descriptive only.
- Validation criterion passed: 18/24 noise errors flagged (vs 6/24
  confidence-only); CBW 17/21; burden 28/1000 (false-alert 2.7%).
- Locked test: 25/27 flagged (0.926) vs 5/27 (0.185); CBW 24/25 (0.96);
  burden 36/1000 (false-alert 3.3%). Does not modify predictions or
  system_state (contract change deferred as a PI decision).

## ROB-001 — robustness variant (non-default)

- Single-factor change vs SEG-001: train-only Gaussian noise σ=0.05.
  Checkpoint `checkpoints/ROB-001/best.pt` (`36091d63…`).
- Validation: noise degradation eliminated (SEG −0.173 → ROB +0.006) with
  clean cost (mean −0.045, p10 −0.170). Locked test reproduces the trade-off
  (mean −0.041): ROB-001 Dice 0.8207 vs SEG-001 0.8617; disagreement
  9/856 = 1.05%. H-ROBUST-001 supported; ROB-001 is not an unconditional
  improvement and does not replace SEG-001.

## Calibration, localization, deployment

- Calibration: T=0.5116, τ1=0.95, τ2=0.05 (validation-only, frozen).
  Reliability verified on test (ECE 0.002655 recomputed == stored == locked).
- Localization: threshold 0.5, no morphology; bbox xyxy ints in original
  pixels, centroid of largest component, area = sum of components
  (`configs/segmentation/localization.yaml`, `17217011…`).
- Deployment: FP16/autocast (val floors Δ 0.0 / −0.000029, deterministic);
  ONNX accurate-but-no-faster (closed); K=8 batching exact-but-slower
  (rejected); quantization not opened. T4 FP16 medians: classify 0.0125s /
  segment 0.0178s / K8 0.318s / analyze 0.0301s; 370/578MB. CPU fallback
  verified; 28-test CI green.

## Known limitations (summary; see Limitations & Validation Report)

Cross-split contamination (disclosed, sensitivity-verified); no subject-level
independence claim; 140 classification-only healthy cases without seg GT;
noise sensitivity (PBA-003); glioma lower-tail weakness; OOD detection not
claimed; single-release (BRISC 2025) validity only. Research prototype —
not a clinical device.
