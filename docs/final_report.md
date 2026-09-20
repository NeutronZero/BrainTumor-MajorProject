# Final Project Report — Verified Brain-Tumor MRI Analysis Platform

Research prototype — not clinical. All numbers are measured results under
stated frozen regimes (evidence paths in parentheses), never diagnostic
accuracy claims.

## 1. Requirements

Eleven functional + eight non-functional requirements (`docs/requirements/`),
frozen under ENG-001 with zero open gaps: calibrated 4-class classification,
threshold-0.5 segmentation with original-pixel localization, descriptive
consistency/reliability/quality/explanation observers, 9-endpoint API,
direct-call UI, frozen system-state precedence, reproducibility,
determinism, CPU fallback, FP16 deployment, offline operation.

## 2. Architecture

Single shared InferenceService; FastAPI thin wrapper; Streamlit direct
calls. Pipeline: bytes → quality → classification → calibration →
segmentation → localization → consistency → reliability → system state →
API/UI. Diagrams: `docs/architecture/system_flow.png`,
`docs/architecture/release_chain.png`. ROB-001 lives outside the default
path as a documented variant.

## 3. Implementation

CLS-001 ConvNeXt-Tiny (seed 42, CE ls=0.1, AdamW cosine, best epoch 14/30);
SEG-001 vanilla U-Net (Dice+BCE, epoch 27/30, norm from seg-train);
frozen T=0.5116, τ1=0.95, τ2=0.05 (validation-only); threshold 0.5, no
morphology; UNC-001 K=8 σ=0.05 (seeds 7003+k); Q01–Q09 fixed constants;
Grad-CAM hooks on `features.7.2.block.0`; REL-001 fixed stable/review rule.

## 4. Verification

Gate-0 with disclosed conditional pass (7 cross-split identities; N=993
sensitivity deltas 3.5e-05/2.5e-05); PBA-001..005 failure analysis;
49-test CPU CI (synthetic only); V&V-001 five gates PASS; OFF-001 socket-guard
PASS; two independent forensic audits with all findings corrected
(CORRECTION-001); quality gate 0/1000 + 8/9; observer-equivalence proofs.

## 5. Validation (locked)

CLS-001 N=1000: acc 0.9950, macro-F1 0.9952, AUC 0.999936, ECE 0.0027
(`outputs/test_evaluation_7b860dca72ea.json` + addendum). SEG-001 N=860:
Dice 0.8617, median 0.9385, IoU 0.7934; disagreement 1.17% < 5%. ROB-001:
trade-off reproduced (−0.041 locked vs −0.045 val). UNC-001: 0.926
detection at 0.033 burden on locked test. Sensitivity populations confirm
contamination-independence throughout.

## 6. Deployment

FP16/autocast retained (floors Δ 0.0 / −0.000029); ONNX faithful but not
faster (closed); K=8 batching exact but slower (rejected); quantization
never opened. T4 FP16 medians 0.0125/0.0178/0.318/0.0301s; 370/578MB;
repeat-deterministic; CPU fallback verified. Deployment docs:
`docs/deployment/profiles.md`, `docs/user_guide/usage.md`, `docs/api/`.

## 7. Limitations

Noise sensitivity; confidence-only uncertainty gaps (UNC-001 detects most
at 3.3% burden); glioma lower tail; upstream cross-split contamination
(disclosed, never "leakage-free"); no subject independence; 140 healthy
cases without seg GT; single-release validity; no OOD claim; not clinical.

## 8. Conclusion

Requirements → architecture → implementation → verification → validation →
deployment → limitations, each link evidenced and frozen. Release chain:
SB-1 → SB-1.1 → ADVANCED-FOUNDATION → ADVANCED-R1 → VV-001 → ENG-001 →
OFF-001 → CORRECTION-001 → ENG-002. The method — freeze early, verify
obsessively, correct transparently — is the project's principal result
alongside the working system.
