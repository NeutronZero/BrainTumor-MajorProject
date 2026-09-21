# HEADLINE RESULTS & ENGINEERING SUMMARY CARD
## BrainTumor-MajorProject: Multi-Stage MRI Analysis & Reliability Platform
### Authoritative Baseline: Release `FINAL-001` (`48dd3ae`) / Lineage `ad01b73` (`CORRECTION-002`)

---

### GOVERNANCE & STATUTORY STATEMENT
> **Engineering Prototype Notice:** Applied software and computer-vision engineering prototype for an **Electronics and Communication Engineering (ECE)** capstone. Not a medical device; not certified for clinical diagnosis, patient triage, or therapy planning. Evaluated strictly on the retrospective BRISC 2025 dataset (`arXiv:2506.14318`). No claims of patient-level independence, external clinical generalizability, or state-of-the-art superiority are made. The dataset is not claimed to be "leakage-free"; 7 cross-split duplicate hashes are forensically disclosed and sensitivity-tested.

---

### 1. FROZEN CLASSIFICATION BENCHMARK (`CLS-001`: ConvNeXt-Tiny, 27.8M Params)

| Evaluation Metric | Primary Cohort ($N=1000$) | Sensitivity Cohort ($N=993$) | Contamination Delta ($\Delta$) | Primary Cohort 95% Confidence Interval |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | **0.995000** ($995/1000$) | **0.994965** ($988/993$) | $+0.000035$ | **$[0.988371, 0.998375]$** (Exact Clopper-Pearson Binomial) |
| **Macro-Averaged F1** | **0.995167** | **0.995142** | $+0.000025$ | **$[0.990400, 0.999101]$** (10,000-Replicate Percentile Bootstrap) |
| **ROC-AUC (OvR)** | **0.999936** | **0.999936** | $0.000000$ | Point estimate |
| **Expected Calibration Error**| **0.002655** | **0.002657** | $-0.000002$ | Calibrated via $T=0.5116$ (Uncalibrated: $0.00760$) |
| **Uncertain Predictions** | **4** ($0.4\%$) | **4** ($0.4\%$) | $0$ | Trapped by thresholds $\tau_1=0.95, \tau_2=0.05$ |

*Note on Sensitivity Delta:* The $+3.5 \times 10^{-5}$ accuracy difference is an exact descriptive arithmetic delta resulting from deterministic exclusion of the 7 duplicate test cases, not a random variable.

---

### 2. FROZEN SEGMENTATION BENCHMARK (`SEG-001`: Vanilla U-Net, 31.0M Params)

| Metric | Primary Test ($N=860$) | Sensitivity Test ($N=853$) | Engineering Description |
| :--- | :--- | :--- | :--- |
| **Mean Dice Coefficient** | **0.861659** ($\approx 0.8617$) | **0.860962** ($\approx 0.8610$) | Primary continuous overlap |
| **Median Dice Coefficient** | **0.938537** ($\approx 0.9385$) | **0.937901** ($\approx 0.9379$) | Robust central tendency |
| **10th Percentile Dice ($P_{10}$)**| **0.656268** | **0.654030** | Measures lower-tail margin difficulty |
| **Mean Intersection-over-Union**| **0.793411** ($\approx 0.7934$) | **0.792543** ($\approx 0.7925$) | Strict spatial intersection |
| **Empty Predictions** | **10** ($1.16\%$) | **10** ($1.17\%$) | Contained by `tumor_unlocalized` state |
| **Multi-Component Predictions** | **81** ($9.42\%$) | **81** ($9.50\%$) | Detected spatial fragmentation |

*Severe Failure Subgroup:* Axial Glioma ($N=85$, Mean Dice: $0.6770$, Median: $0.7765$, $P_{10}$: $0.0000$, Empty: $5$). Exactly 100% (10/10) of empty masks occur in small lesions ($\le 442\text{ px}$, Quartile 1).

---

### 3. SYSTEM STATE DISTRIBUTION (PRIMARY TEST $N=1000$)

$$\text{degraded} \succ \text{uncertain} \succ \text{tumor\_unlocalized} \succ \text{tumor\_localized} \succ \text{healthy}$$

- `tumor_localized`: **846** ($84.6\%$) — Validated spatial consensus (tumor class + non-empty mask).
- `tumor_unlocalized`: **10** ($1.0\%$) — Cross-model discrepancy containment (tumor class + empty mask).
- `uncertain`: **4** ($0.4\%$) — Sub-threshold confidence ($< 0.95$) or margin ($< 0.05$).
- `healthy`: **140** ($14.0\%$) — Non-tumor class invariant (segmentation bypassed).
- *Discrepancy Rate:* 10 / 856 confident tumor cases = **$1.168\%$** (safely within pre-registered $< 5.0\%$ threshold).

---

### 4. PERTURBATION OBSERVER MATRIX (`UNC-001`, $K=8$ Probes, $\sigma=0.05$ AWGN)

| Observer Status | Perturbation Error (True Error) | Invariant (No Error) | Total Cohort |
| :--- | :--- | :--- | :--- |
| **Flagged ($\alpha < 1.0$)** | **$25$ (True Positive, TP)** | **$11$ (False Positive, FP)** | **$36$** (Flag Rate: $3.60\%$) |
| **Unflagged ($\alpha = 1.0$)** | **$2$ (False Negative, FN)** | **$962$ (True Negative, TN)** | **$964$** |
| **Total Evaluated** | **$27$** | **$973$** | **$1000$** |

- **Detection Efficacy:** Precision $= 69.44\%$, Recall $= 92.59\%$, F1 $= 79.49\%$, FPR $= 1.13\%$.
- **Confident-But-Wrong (CBW) Capture:** Captures **24 of 25** ($96.0\%$) confident errors under noise (vs. softmax thresholding capturing only 5 of 27 / $18.52\%$).

---

### 5. FIVE PERMANENT ARCHITECTURAL FREEZES

1. **Vocabulary:** Exactly `glioma`, `meningioma`, `pituitary`, `notumor`. `tumor_detected` is strictly derived (`predicted_class != 'notumor'`).
2. **Localization Geometry:** Integer bounding box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ in original input pixels; centroid $(c_x, c_y)$ float of largest component; area in pixels. Empty mask yields `bbox=None, centroid=None, area_pixels=0`.
3. **Calibration & Operating Parameters:** Temperature $T=0.5116$, operating thresholds $\tau_1=0.95, \tau_2=0.05$, segmentation threshold $\tau_{\text{seg}}=0.5$. Locked on validation only.
4. **Dataset Partitioning:** Seed 42 partition of 5,000 training pool into 4,000 train / 1,000 val (hash-quarantined); 1,000 official test images locked for single-pass final evaluation.
5. **Unified Service Pattern:** Single shared `InferenceService` instantiated in-process by both FastAPI and Streamlit. Zero duplicate model weights, zero HTTP loopback overhead.

---

### 6. HARDWARE & LATENCY PROFILES (NVIDIA T4 GPU FP16 vs. CPU FP32)

| Stage | NVIDIA T4 / FP16 ($N=60$) | x86_64 CPU / FP32 | Evidence Source |
| :--- | :--- | :--- | :--- |
| **Classifier (`classify`)** | **$12.5\text{ms}$** ($P_{95}: 13.8\text{ms}$) | $\approx 220\text{ms}$ | `outputs/SYSINT/fp16_check.json` |
| **Segmenter (`segment`)** | **$17.8\text{ms}$** ($P_{95}: 18.0\text{ms}$) | $\approx 2000\text{ms}$ | `outputs/SYSINT/fp16_check.json` |
| **Core Analysis (`analyze`)** | **$30.1\text{ms}$** ($P_{95}: 31.4\text{ms}$) | $\approx 1400\text{ms}$ | `outputs/SYSINT/fp16_check.json` |
| **Consistency Probes ($K=8$)**| **$318.2\text{ms}$** ($P_{95}: 336.4\text{ms}$)| $\approx 2100\text{ms}$ | `outputs/SYSINT/fp16_check.json` |
| **Peak VRAM Allocated** | **$370.1\text{ MB}$** | N/A | `outputs/SYSINT/fp16_check.json` |
| **Peak VRAM Reserved** | **$578.0\text{ MB}$** | N/A | `outputs/SYSINT/fp16_check.json` |

---

### 7. ECE ENGINEERING CURRICULUM PILLARS

| Domain | Implemented Component | Key Technique / Formulation |
| :--- | :--- | :--- |
| **1. Digital Image Processing (DIP)** | Pipeline Resampling, Localization Engine | Forward/inverse bilinear scaling; connected components; spatial moments |
| **2. Digital Signal Processing (DSP)** | ConvNeXt-Tiny Stem, Quality Gate (`gate.py`) | 2D discrete FIR convolution ($7 \times 7$ depthwise); moment estimation ($\bar{x}, s^2$) |
| **3. Probability & Random Processes** | Perturbation Observer, Calibrator | AWGN modeling ($\sigma=0.05$); Platt scaling ($T=0.5116$); Clopper-Pearson CIs |
| **4. Communication & Networks** | FastAPI Endpoints, Ingestion Guards | Multipart serialization; HTTP/1.1 contracts; socket interception (`OFF-001`) |
| **5. Computer Architecture** | Precision Profiles, Hardware Runtime | CPU SIMD vs. GPU Tensor Cores; FP16 mixed precision; VRAM management |
| **6. Numerical Systems & Computing** | Numerically Stable Softmax, Determinism | Log-sum-exp formulation; IEEE 754 precision; algorithmic determinism controls |
| **7. Fault-Tolerant Instrumentation** | State Precedence, Defensive Quality Gates | Priority state machine; defensive screening (`Q01`–`Q09`); contract validation |

---
*End of Headline Results Summary — BrainTumor-MajorProject (Release FINAL-001 / Lineage ad01b73)*
