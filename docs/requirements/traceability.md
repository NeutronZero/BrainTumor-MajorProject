# Traceability Matrix — ENG-001 Baseline (frozen)

Requirement → Design component → Implementation → Verification evidence.
VV-001 evidence is referenced, not regenerated. No locked-test contact.

| Req | Design | Implementation | Evidence |
| --- | --- | --- | --- |
| FR-CLS-1/2 | Calibration + state branch | service.classify | CI classify/contract tests; locked N=1000 (0.9950/0.9952) |
| FR-SEG-1 | U-Net + thr 0.5 | service.segment | CI segment tests; locked N=860 (0.8617) |
| FR-LOC-1 | extract() orig-px | service.localize | CI invariants + geometry gate; 785-pair invariants |
| FR-CON-1 | K=8 rule | service.consistency | CI determinism/equivalence; UNC-001 val + locked |
| FR-REL-1 | stable/review rule | service.reliability + engine | CI combinations; REL-001 evidence |
| FR-QLT-1 | Q01–Q09 codes | service.quality + gate.py | CI codes; QUALITY (0/1000, 8/9) |
| FR-EXP-1 | observer + overlays | service.explain + gradcam/report | CI equivalence/determinism; EXPL-001 |
| FR-API-1 | 9 endpoints + envelopes | app/api/main.py | integration tests (415/413/422/500) |
| FR-UI-1 | direct-call rendering | app/streamlit/app.py | payload tests + live checks |
| FR-SYS-1 | precedence + invariants | derive_system_state + validators | contract tests + gate truth table |
| NFR-REPRO-1 | manifest regen | build_manifest.py | manifest + clean-clone validation |
| NFR-DET-1 | eval mode, fixed seeds | service (all paths) | determinism tests (CPU + T4 probe) |
| NFR-CPU-1 | device-gated autocast | service._autocast | CPU CI suite |
| NFR-GPU-1 | FP16 autocast | same | T4 floors |
| NFR-PERF-1 | bench suite | scripts/bench | FP32 ref + FP16 medians + memory |
| NFR-PORT-1 | anchored ignore, skips | .gitignore, needs_data | clean-clone (47+2) |
| NFR-OFF-1 | offline-first intent | service/app + scripts/off/verify_offline.py | EXECUTED: OFF-001 PASS (guard effective, startup/load/inference/API offline, zero external calls); loopback-exempt per docs/off/OFF-001.md |
| NFR-ERR-1 | typed envelopes | API + Streamlit branches | envelope tests |

Gaps: none open. NFR-OFF-1 closed by OFF-001 (executed evidence).
