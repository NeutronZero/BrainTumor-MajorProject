# System Architecture — ENG-001 Baseline (frozen)

Single execution path: the shared InferenceService. FastAPI is a thin
wrapper; Streamlit direct-calls the service (no HTTP loopback).

```text
Input bytes
   ↓
Quality gate (Q01–Q09, descriptive)
   ↓
Classification (CLS-001, frozen T/τ)
   ↓
Calibration (applied, frozen)
   ↓
Segmentation (SEG-001, thr 0.5) ──┐ tumor predictions only
   ↓                               │
Localization (extract, orig px)    │
   ↓                               │
Consistency K=8 (UNC-001) ─────────┘ descriptive branches
   ↓
Reliability (REL-001 stable/review, descriptive)
   ↓
System State (frozen precedence)
   ↓
API / Streamlit (contract-validated outputs)
```

## Components (all frozen unless noted)

| Component | Implementation | Frozen record |
| --- | --- | --- |
| Quality | src/brain_tumor/quality/gate.py | QUALITY validation |
| Classifier | ConvNeXt-Tiny, ckpt hash 0fa58033 | locked N=1000 |
| Calibration | T=0.5116, τ1=0.95, τ2=0.05 | calibration_frozen.json |
| Segmenter | U-Net, ckpt hash ce29df5e | locked N=860 |
| Localization | extract(), thr 0.5, min-area 10 | 785-pair invariants |
| Consistency | K=8, σ=0.05, seeds 7003+k | UNC-001 val + locked |
| Reliability | engine.py stable/review rule | REL-001 evidence |
| Explanation | Grad-CAM features.7.2.block.0 | EXPL-001 evidence |
| Service/API/UI | service.py, main.py, app.py | 49-test CI + live checks |
| Deployment | FP16/autocast CUDA-gated | T4 floors + medians |

ROB-001 (noise U-Net) is a documented non-default variant, outside the
default path. No component may be substituted without a new workstream.
