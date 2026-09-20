# V&V-001 Verdict: PASS

All five gates executed against ADVANCED-R1 (92eb2f0) with zero failures.
No locked-test contact; synthetic inputs only. ADVANCED-R1 itself unmodified
(all V&V artifacts live on branch vv-001 + outputs/VV/).

## Gate results

1. **Fresh environment** — pristine GitHub clone (240 files, ADVANCED-R1):
   24/24 static + weight-free checks pass; LFS OIDs match the manifest
   (blobs not re-pulled: quota rationale recorded; bytes proven identical
   by hash on the local leg).
2. **Inference contracts** — labels/order, T=0.5116, τ1/τ2, threshold 0.5,
   original-pixel units, precedence table, empty/healthy paths, K=8/σ/seeds,
   quality codes, reliability non-override, explain equivalence, extract
   geometry: all pass on hash-identical local weights.
3. **Cross-interface** — service == API == Streamlit on identical input.
4. **Deployment** — CPU load/inference/determinism live; CUDA leg upheld by
   recorded T4 evidence (FP16 floors, ONNX split verdict, batching
   rejection all reproduce from their JSONs). No new kernel, no optimization.
5. **Evidence integrity** — checkpoint hashes + locked CLS/SEG/sensitivity/
   UNC/ROB spot values all match.

## Artifacts

outputs/VV/: gate1_clone.json, gate2_local.json, gate3_cross_interface.json,
gate45_deploy_evidence.json. Scripts: scripts/vv/gate_fresh_contracts.py,
gate_cross_interface.py, gate_deploy_evidence.py.

## Standing notes

- LFS bandwidth quota (~792MB already consumed) constrains future full
  re-pulls; OID+hash identity is the established substitute.
- PowerShell `>` redirection writes UTF-16; archived JSONs were converted
  to UTF-8 at archive time.
