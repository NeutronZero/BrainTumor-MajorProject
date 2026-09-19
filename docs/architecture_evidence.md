# Architecture Evidence Record (frozen before CLS-001)

Status: EVIDENCE, not selection. No checkpoint, preprocessing, normalization,
augmentation, split, or hyperparameter is inherited. All numbers below are
prior-project results on the prior processing regime, cited for candidate
prioritization only. They are not results of this project.

## 1. Candidates explicitly supported by reconstructed evidence
- ConvNeXt-Tiny (prior C4)
- Swin-Tiny (prior C5)
- DenseNet-121 (prior C3)
- ResNet-50 (prior C1)
- EfficientNet-B0 (prior C2)
- Custom scratch CNN (prior C0, negative control)

## 2. Prior evidence (controlled C0–C5 x seeds 42/43/44, frozen val-Macro-F1)
- Val Macro-F1: C4 0.9967, C5 0.9950, C3 0.9945, C1 0.9925, C2 0.9903, C0 0.66
- Test (N=1000, locked reporting): C4 Macro-F1 0.9953 / Acc 0.9947 / ECE 0.0040
- Sensitivity N=993: order identical, max |d| 0.0012
- Sources: `braintumor/docs/09-results.md`, `Major Project/research/SOURCES.md`

## 3. Statistical caveat
- C4 vs C3 seed-stratified: McNemar exact p 0.727/1.00/1.00, permutation
  p 0.72/1.00/1.00 — ranking is descriptive; C4/C3/C5 indistinguishable at n=3.
- Shared hyperparameters: controlled comparison, not intrinsic superiority.
  No "SOTA/best model" claim is supported.

## 4. Transferability caveat
- Prior regime: prior letterbox pipeline, prior dev split (3997/1003), CE no
  weighting, AdamW 3e-4 cosine, patience 7 on val Macro-F1.
- This project: fresh manifest/split (4000/1000, different membership), own
  preprocessing decision (configs/preprocessing/), CE label-smoothing 0.1
  (plan default), own training run. Prior numbers must not be compared
  numerically with ours.

## 5. EfficientNet-B0 evidence
- 5th of 6 on prior validation (0.9903), within thousandths of C1/C5.
- Sole large Tier-4 perturbation effect: Gaussian sigma 0.01 drove C2
  0.9902 -> 0.2225 while C1/C3/C4/C5 shifted <=0.012 (descriptive, mechanism
  undescribed). Caution against default selection.
- Legacy (non-BRISC) selection of E-B0 belongs to a different data regime and
  carries zero weight here. E-B0 is NOT the default.

## 6. Decision for CLS-001
- CLS-001 architecture: **ConvNeXt-Tiny** (ImageNet-pretrained init as explicit
  transfer-learning decision, plan S33; 4-output head;-RNG seed 42).
- Rationale: descriptive rank 1 on the only BRISC controlled comparison
  available, with caveats 3–4 recorded. Status: evidence-based baseline for
  THIS project to confirm or refute — not a crowned winner.
- Prioritized comparator if CLS-001 warrants follow-up: DenseNet-121
  (rank 3, statistically tied, cheaper than Swin). Trigger: CLS-002 proposal
  with its own question/hypothesis; never silent.
- SEG-001 architecture: **vanilla U-Net** per plan S43 (independent resolution/
  norm/aug/training decisions in configs/experiment/SEG-001.yaml).
