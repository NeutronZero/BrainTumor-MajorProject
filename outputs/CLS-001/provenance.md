# CLS-001 output provenance (honest accounting)

Research prototype — not a medical diagnosis. This file records what is and
is not recoverable for the CLS-001 experiment directory.

## What is committed here

- `calibration_frozen.json` only — the frozen temperature-scaling and
  operating-point parameters (T=0.5115954875946045, tau1=0.95, tau2=0.05),
  selected on validation only and locked.

## What is NOT recoverable

- The original `metrics.json` and `run_config.json` produced by the Kaggle
  training kernel were never persisted into this repository and are
  unrecoverable. They are intentionally left absent rather than
  reconstructed or fabricated. (`outputs/CLS-001-smoke/` contains a separate
  1-epoch CPU smoke run and is not a substitute.)

## Checkpoint lineage

- `checkpoints/CLS-001/best.pt` is the C3 state-only release derivative
  (`451e4fc4…`) derived from the original training output (`0fa58033…`) via
  `scripts/release/strip_cls_checkpoint.py`. The derivation removes optimizer/
  scheduler/scaler/RNG keys only; `epoch`, `val_metric`, and an in-file
  `lineage` record are preserved, and inference was proven bit-identical on a
  3-case battery at derivation time. Locked-test evaluation attribution
  remains the original checkpoint (documented lineage, not a silent swap).

## Where the locked-test metrics live

- All locked N=1000 primary and N=993 sensitivity classification metrics are
  reported in `outputs/test_evaluation_7b860dca72ea.json` (plus the AUC
  addendum `outputs/test_evaluation_7b860dca72ea_addendum_auc.json`).
- Per-case locked predictions: `outputs/PBA-001/per_case_test.json`.
- Headline confidence intervals: `outputs/PBA-001/bootstrap_ci.json`
  (producer: `scripts/evaluate/bootstrap_ci.py`).
