# Dataset provenance — Gate 0 rev A + PI CONDITIONAL-PASS (2026-09-15)

- dataset: BRISC 2025 (Fateh et al., arXiv:2506.14318), Kaggle
  `briscdataset/brisc2025`, CC BY 4.0, local bytes `data/brisc2025`
- manifest SHA256 recomputed equal: csv `4eaa0c3b88ce1fd246d90382c1dcd1010cc126fff1bd3a4a415159661ca8b605`,
  json `ada767e102e9e7ee788b50188712dfe19ba825c250c362d831b560902c249e3a`
- classification: 6000 = train 5000 (glioma 1147, meningioma 1329,
  no_tumor 1067, pituitary 1457) + test 1000 (254/306/140/300)
- segmentation: 4793 pairs (train 3933 / test 860), `binary_intent_aa`
- contamination: 7 cross-split image hashes (9 train + 7 test files), disclosed;
  primary test N=1000 locked, sensitivity N=993 locked
- project split: 4000/1000 frozen (`outputs/data_gate_0/project_split.json`)
- remote verification (2026-09-15, Kaggle CLI, read-only): owner/slug
  `briscdataset/brisc2025`, CC BY 4.0, 15,591 files, per-class counts match,
  both manifest SHAs recomputed equal; single 2025-11-04 snapshot; journal
  version cites Scientific Data doi:10.1038/s41597-026-06753-y. No contradiction
  with frozen identity. Scratch files deleted after verification.
- evidence: `outputs/data_gate_0/` (summary, duplicate, mask, split-overlap,
  forensic, manifest, exclusion list); decision: `docs/gate0_conditional_pass.md`
- Limitation: reliable patient/subject identifiers are insufficient to guarantee
  complete subject-level independence. Complete patient-level independence
  between train and test cannot be claimed.
