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

## Round-2 audit (UI pass + hardening re-check)

The Streamlit UI was rebuilt (sidebar flow, session persistence, styled
result card, theme config) and a second audit pass was run over the whole
surface. Findings, all fixed:

| # | Finding | Fix |
|---|---|---|
| 1 | `Dockerfile` CMD used `app.api.main:app` together with `--app-dir app/api` — contradictory module resolution, container would crashloop at boot | CMD → `main:app --app-dir app/api` (the import root is `app/api`; `main.py` self-bootstraps `src/`) |
| 2 | `python-multipart` installed in dev env but absent from lock — **every upload endpoint would 500** in a fresh container/clean venv | Declared in `pyproject.toml` + pinned in `requirements.lock` (0.0.32); proven by a real multipart upload returning 200 in the clean-venv boot test |
| 3 | `/health` rehashed ~600MB of artifacts on every call — unusable as a liveness probe | 300s TTL cache (`_integrity_report_cached`); full rehash remains available via `build_manifest.py` |
| 4 | `/analyze_batch` empty-batch exit bypassed metric accounting | Covered by middleware-level accounting (see #5) |
| 5 | REL-002 accounting rule was unenforceable: FastAPI validation rejections (422) never reach handlers, so the in-handler scheme under-counted by design | `INFERENCE_REQUESTS` moved to `RequestIDMiddleware` (single accounting point at terminal HTTP status; 35 in-handler call sites removed); regression tests added for validation-422 and non-inference-path exclusion |

Verification after fixes: full suite **113 passed** (48 unit/data/holdout +
65 regression/integration), ruff clean, headless Streamlit boot HTTP 200.

## Container-equivalent boot proof (daemon-free)

Docker Desktop on this machine is stuck in "starting engine" and was left
as-is per user instruction. To avoid inventing a container PASS, the
container's runtime contract was verified daemon-free instead: the clean
venv (installed purely from `requirements.lock`) booted the API with the
**exact container CMD** (`python -m uvicorn main:app --app-dir app/api`) and
served: `/health` → `ok` (both models loaded, calibration + integrity
present), a real multipart `/classify` upload → HTTP 200,
`/metrics` → exactly one `inference_requests_total` observation. This proves
the image's dependency set and startup command; it does NOT prove the image
builds.

## Release status (final, round-2 closure)

> **REL-002 application/runtime hardening: COMPLETE.**
> **Regression evidence: 113/113 PASS.**
> **UI integration: VERIFIED.**
> **Dependency/startup/multipart behavior: VERIFIED in clean environment.**
> **Container build: UNVERIFIED — Docker daemon unavailable (stuck starting).**
> **Release remains CONDITIONAL; no container PASS claimed.**

Held here until Docker is operational. The container gate, when run, covers
container-specific behavior only (build → run → /health → multipart
/classify → /metrics accounting → shutdown → restart); the 113-test suite
does not need to be rerun for it. No further application-code changes are to
be made to compensate for the unavailable daemon.

When Docker becomes available, verify against this exact sequence:

```text
docker build
    ↓
docker run (checkpoints/outputs mounted read-only)
    ↓
/health → "ok" (both models, calibration, integrity present)
    ↓
multipart /classify → 200
    ↓
/metrics → inference_requests_total observations present
    ↓
shutdown
    ↓
restart
```

Uncommitted work awaiting scoped commits (provenance rules as REL-002):
round-2 fixes (`app/api/main.py`, `pyproject.toml` deps hunk,
`requirements.lock`, `Dockerfile`, `tests/regression/test_rel002_hardening.py`)
and the UI pass (`app/streamlit/app.py`, `.streamlit/config.toml`), plus the
pre-existing Phase-E tree.
