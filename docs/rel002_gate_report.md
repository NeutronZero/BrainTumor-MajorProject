# REL-002 Gate Report — Production Boundary Hardening

**Status: CONDITIONAL PASS.**
Functionally green and committed with strict per-scope provenance; the
container build remains **UNVERIFIED (infrastructure/tooling unavailable)**.
SB-1 remains immutable at `0f496a6`.

## Commit structure (provenance)

REL-002 is intentionally split across two commits, each single-scope:

1. `73c4396` — `phase-e(api): capture pre-REL-002 InferenceService wrapper
   state`. Verbatim pre-REL-002 `app/api/main.py` (metrics via direct
   `.labels()` calls, `/analyze_batch` unbounded). This commit exists so the
   REL-002 diff on that file is pure: it modifies only what REL-002 changed.
2. REL-002 commit — the hardening delta only:
   `app/api/main.py` (106+/16−, `_count_request` centralization + batch
   gates), `pyproject.toml` (declared deps hunks ONLY — the Phase-E `dev`
   extras section is excluded), `requirements.lock`, `Dockerfile`,
   `.dockerignore`, `docs/deployment/profiles.md` (REL-002 sections only),
   `docs/rel002_gate_report.md`,
   `tests/regression/test_rel002_hardening.py`.

The remaining ~85 dirty Phase-E files (data-gate evidence, bench scripts,
streamlit, docs, candidate-tracking tests, `src/brain_tumor/models/`,
`locked_holdout.py`, …) are out of REL-002 scope and stay in the worktree for
their own scoped commit(s).

## Gates

| Gate | Status | Evidence |
|---|---|---|
| Dependency reproducibility | **PASS** | `filetype==1.2.0`, `prometheus-client==0.26.0` declared; lock regenerated from verified env. The old lock was **unresolvable** (`streamlit==1.50.0` requires `pandas<3` vs pinned `pandas==3.0.3`) — caught by the clean-venv install, not assumed. `pip check` → "No broken requirements found". Load-bearing transitives `starlette==1.2.1` + `httpx==0.28.1` pinned so fresh-venv TestClient transport matches the verified env. |
| Clean import | **PASS** | Fresh venv → install from lock → app import OK (15 routes, both checkpoints loaded). |
| Batch resource bounds | **PASS** | 8 files max, 32MB aggregate declared bytes, 64MP aggregate decoded pixels. Count/byte gates proven pre-parse (test asserts `_read_bounded` never called when the gate trips); pixel gate proven pre-inference (spy). |
| Observability consistency | **PASS** | All 9 inference endpoints report exactly one `INFERENCE_REQUESTS` observation per request via `_count_request`; static test forbids direct `.labels()` outside the helper. |
| Container build | **UNVERIFIED** | Docker artifacts written and statically reviewed; daemon unresponsive on the build machine (`docker build`, even `docker images`, hang). NOT a PASS. Verification checklist when daemon is available: `docker build` → `docker run` with ro-mounted checkpoints/outputs → `/health` returns `ok` → basic `/classify` inference → shutdown/restart. |
| Targeted regression | **PASS** | `tests/regression/` 37 passed (incl. 7 new REL-002 tests), run in the clean venv. |
| Full regression | **PASS** | 111 passed, 0 failed (~6:53, verified env). `reproducibility_check: OK`. |
| Manifest/hash verification | **PASS (not committed)** | `build_manifest.py` → "missing=none". The rebuilt `docs/release_manifest.md` + `outputs/data_gate_0/*` hashes reflect the **Phase-E worktree**, not the REL-002 commit, so they are deliberately excluded from the REL-002 commit; they belong to the Phase-E snapshot commit that follows. |

## Release status

```
REL-002 implementation:        PASS
REL-002 regression/evidence:   PASS
REL-002 container verification: UNVERIFIED (tooling unavailable)
REL-002 release:               CONDITIONAL — container build must be
                               executed and pass the checklist above
                               before any container-based deploy
```
