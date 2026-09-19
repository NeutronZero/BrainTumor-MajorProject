# Data card — Gate 0 rev A + PI CONDITIONAL-PASS (2026-09-15)

- name: BRISC 2025; source: Kaggle `briscdataset/brisc2025`; license: CC BY 4.0;
  manifest SHA256: `4eaa0c3b…ca8b605` (csv), `ada767e1…c249e3a` (json)
- class counts: train — glioma 1147, meningioma 1329, no_tumor 1067,
  pituitary 1457; test — 254/306/140/300
- planes: axial/coronal/sagittal codes present in filenames; plane metadata is
  released, not verified ground truth (13 plane-conflicted groups retained, never
  relabeled); per-stratum membership in `project_split.json`
- masks: 4793 PNG, single-channel, binary-intent with antialiased edges
  (intermediate fraction 0.0034); convention `>127 -> 1` documented, files
  untouched; no `no_tumor` masks; 0 empty; 21 divergent-mask images flagged
- resolution: heterogeneous (min 174x195, max 1365x1427); stats:
  outputs/data_gate_0/dataset_summary.json
- dataset-author preprocessing: grayscale T1 slices with letterbox-style
  framing per prior-study provenance (our preprocessing NOT selected yet)
- representativeness/scanner/geography: unverified — not claimed
- contamination: 7 cross-split image hashes disclosed; N=1000 primary locked,
  N=993 sensitivity locked; 9 contaminated train files retained+flagged in
  project-train, never validation
- Residual subject-level leakage limitation: complete patient-level
  independence cannot be claimed.

## Locked-evaluation record (appended post-validation; no data changed)

- Development split (frozen): project train 4000 / val 1000 from the
  5000-image train pool; manifest `outputs/data_gate_0/project_manifest.csv`
  (`c120e5bf…`). Official test 1000 (`test_lock.yaml`, `d46624af…`).
- Classification: locked N=1000 primary + N=993 sensitivity (7 excluded
  cross-split identities); deltas negligible (3.5e-05/2.5e-05).
- Segmentation: 4793 Gate-0-verified pairs; seg-train 3148 / seg-val 785
  (val = seg-train images whose stems map to project val);
  locked seg-test 860 pairs + N=853 sensitivity (delta 0.0007).
- Audits: exact-duplicate, near-duplicate, and split-overlap reports in
  `outputs/data_gate_0/`; 21 divergent-masks flagged; plane-conflicted
  groups retained without relabeling.
- Healthy-only: 140 classification-test notumor cases have no segmentation
  ground truth — unmeasured for segmentation, never treated as zero outcomes.
