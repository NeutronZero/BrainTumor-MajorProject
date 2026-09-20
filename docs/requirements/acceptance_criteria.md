# Acceptance Criteria — ENG-001 Baseline (frozen)

A future system state accepts ENG-001 iff every criterion below holds.
Criteria are measurable and refer to frozen artifacts; none requires new
test-set contact.

## Functional acceptance

- AC-CLS: locked N=1000 accuracy 0.9950 / macro-F1 0.9952 / ECE 0.0027
  reproduced from `outputs/test_evaluation_7b860dca72ea.json` (read-only).
- AC-SEG: locked N=860 mean Dice 0.8617 / IoU 0.7934; disagreement 1.17% < 5%.
- AC-CAL: T=0.5116, τ1=0.95, τ2=0.05 byte-identical to calibration_frozen.json.
- AC-UNC: locked detection 25/27 at burden ≤ 4% (recorded 0.926 / 0.033).
- AC-QLT: 0/1000 clean-val false rejects; ≥ 8/9 corruption battery.
- AC-EXP: explain outputs bit-identical to analyze; heatmaps deterministic.
- AC-REL: stable/review rule matches engine spec on the fixture matrix.
- AC-API: all 9 endpoints respond with contract-valid payloads; envelopes
  415/413/422/500 verified.
- AC-UI: Analyze + Explain render all fields; empty/error branches render.
- AC-SYS: precedence truth table passes; validators reject contradictions.

## Non-functional acceptance

- AC-REPRO: manifest regenerates with zero MISSING and hash agreement.
- AC-DET: repeated identical inputs give identical outputs (CPU).
- AC-CPU: full suite passes without CUDA.
- AC-GPU: FP16 floors hold (Δ 0.0 / −0.000029 or better on re-run).
- AC-PERF: T4 FP16 medians within 2× of recorded reference (shared-host
  variance allowance) OR variance documented with same-run interleaving.
- AC-PORT: clean clone reconstructs; dataset-absent tests skip with reason.
- AC-ERR: every failure fixture yields a typed error, never a traceback.

## Status

AC-OFF (executed offline proof) is deferred as the single documented gap.
All other criteria are satisfied at ENG-001 baseline commit.
