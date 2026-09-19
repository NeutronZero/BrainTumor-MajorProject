# GATE-0 PI DECISION: CONDITIONAL-PASS (contamination disclosed)

Date (UTC): 2026-09-15. Authority: PI verdict. Gate tooling rev A.
Gate 0 strict decision remains FAIL (cross-split contamination); the PROJECT
proceeds under this pre-specified contamination-aware protocol. No threshold
was weakened; `outputs/data_gate_0/gate_decision.json` is preserved as-is.

## Frozen facts
- Dataset: BRISC 2025 (Kaggle `briscdataset/brisc2025`, CC BY 4.0,
  arXiv:2506.14318). Manifest SHAs recomputed equal:
  csv `4eaa0c3b…ca8b605`, json `ada767e1…c249e3a`.
- Classification: 6000 = official train 5000 (1147/1329/1067/1457
  glioma/meningioma/no_tumor/pituitary) + official test 1000 (254/306/140/300).
- Segmentation: 4793 pairs (train 3933 / test 860), `binary_intent_aa`
  (intermediate fraction 0.0034, convention `>127 -> 1` documented, files
  untouched), pairing intact, 0 dim mismatches, 0 empty, 21 divergent-mask images.
- Cross-split contamination: 7 underlying byte-SHA256 image identities shared
  classification train <-> test (9 train files, 7 test files); same 7 images
  span segmentation train/test; 0 mask cross-split duplicates.
- Forensic reconciliation: Revision A finds 35 within-train SHA groups / 38
  excess; prior reconstruction reported 34/36 with the 04176/04177/04178
  triplet tracked separately. 34+1=35 groups, 70+3=73 records, 36+2=38 excess;
  overall unique count identical (5950). Difference is grouping-definition,
  not a dataset difference. Triplet = one SHA group (`3e802dd2…`, pituitary x3)
  that also bridges cross-split. Discrepancy closed, non-blocking.

## Protocol
- Primary locked test N=1000 (official release, contamination disclosed) stays
  principal. Sensitivity locked test N=993 (7 hashes excluded) is sensitivity
  only — never called clean. Exclusion: `cross_split_exclusion_list.json`.
- Official train pool 5000 retained whole; 9 contaminated train files flagged
  `cross_split_contaminated=1`, pre-assigned to project-train, never validation.
- Project split 4000/1000: SHA-atomic units, class x plane strata (class-only
  fallback), seed 42, exact subset-sum DP. Manifest + split hashed
  (`project_manifest.csv`, `project_split.json`).
- Test NEVER used for selection/tuning/thresholding. Subject-level
  independence NOT claimed. External validation NOT claimed.

## Still prohibited
NO model training, NO hyperparameter tuning, NO preprocessing selection,
NO architecture selection, NO test-driven decisions.
Next authorized: architecture evidence record, then CLS-001 (+SEG-001).
