# Engineering Limitations & Validation Report — SB-1

Follows requirements → architecture → implementation → verification →
validation. Reconstructed from frozen evidence only; no test-set contact was
used for documentation. Research prototype — not clinical. Nothing here
establishes clinical validity or generalization beyond the BRISC 2025 release.

## 1. Requirements and architecture

Requirements: 4-class MRI slice classification with calibrated uncertainty;
tumor segmentation with localization (bbox/centroid/area); explicit failure
signaling; CPU/GPU deployability. Architecture: ConvNeXt-Tiny classifier,
vanilla U-Net segmenter, frozen T/τ calibration, extract() localization,
UNC-001 consistency layer (descriptive), FastAPI + Streamlit over one shared
InferenceService under a frozen pydantic contract.

## 2. Implementation evidence

Configs, checkpoints, and preprocessing versions are pinned by hash in the
Release Manifest. Training: CLS-001 (T4, best epoch 14/30, val F1 0.9950),
SEG-001 (T4, epoch 27/30, val micro-Dice 0.8731 / macro 0.8518), ROB-001
(T4, noise σ=0.05, val micro-Dice 0.8337). Calibration and thresholds fit on
validation only. Preprocessing factories shared between train and inference.

## 3. Verification (pre-test)

Dataset verification (Gate 0 + conditional pass); real contamination found
and forensically reconciled (7 cross-split identities + 9 flagged train
files); immutable manifest/split; architecture evidence with statistical
caveats; CPU pipeline smoke runs; PBA-001..005 failure characterization
(5 CLS errors with 2 confident-but-wrong; glioma seg weakness 0.70 val;
ECE 0.002655 independently recomputed; joint denominator 860 matched + 140
unmeasured healthy; plan-§68 perturbation suite showing noise as the dominant
stressor: CLS −0.019 with 20/24 CBW, SEG −0.169 with empty 10→163).
28-test CPU CI (contracts, endpoints, determinism, error envelopes, degraded
paths, Streamlit payload) green; Streamlit live-byte path demonstrated.

## 4. Validation (locked)

- CLS-001: acc 0.9950, macro-F1 0.9952, AUC 0.999936, ECE 0.0027 (N=1000);
  N=993 sensitivity essentially unchanged.
- SEG-001: mean Dice 0.8617, median 0.9385, IoU 0.7934 (N=860); N=853
  Dice 0.8610; disagreement 1.17% (< 5%).
- ROB-001: trade-off reproduced on locked test (−0.041 vs −0.045 val);
  documented variant, not default.
- UNC-001: validation criterion passed (0.75 vs 0.25 detection, 2.7%
  burden); locked test stronger (0.926 vs 0.185, 3.3% burden). Descriptive
  layer; system_state contract unchanged.
- Deployment: FP16 floors pass (Δ 0.0 / −0.000029); ONNX faithful but not
  faster (closed); K8 batching exact but slower (rejected); quantization not
  opened. T4 FP16: analyze 0.0301s median; CPU fallback verified.

## 5. Known failure modes and limitations

1. Gaussian-noise sensitivity (quantified; ROB-001 variant exists with cost).
2. Confidence-only uncertainty misses shifted-input failures (UNC-001 layer
   detects most, not all; 3.3% false-alert burden).
3. Glioma lower-tail segmentation weakness (test p10 0.656; ROB-001 p10 0.544).
4. Multi-component fragmentation increases under noise-trained variant.
5. Cross-split image-identity contamination in the released data (disclosed;
   sensitivity analyses show immaterial aggregate effect, but the release
   itself is imperfect).
6. No subject-level independence claim (slice-level data, identity mapping
   incomplete).
7. 140 healthy classification cases lack segmentation ground truth.
8. Heterogeneous resolutions; plane codes released but unverified.
9. OOD/shift detection beyond σ=0.05 Gaussian noise: not established.
10. Single dataset, single release: no hospital/scanner/population
    generalization claim. Not a clinical device.

## 6. Claim boundary

Every number in this report is a measured result under the stated frozen
evaluation regime (named populations, fixed code, pinned hashes). The
99.50% classification accuracy is a BRISC locked-test measurement, not a
diagnostic-accuracy claim. Deployment latencies are T4-specific medians.
No statement here supports clinical use.
