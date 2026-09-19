# BrainTumor-MajorProject v2.3 — STEP 0/1 scaffold

AUTHORITATIVE plan: `C:\Projects\BrainTumor_MajorProject_Plan.md` v2.3.
Research/educational prototype — **not a clinical diagnostic device**.

## Frozen contracts (§21-25 + 5 freezes)
- classes: `glioma | meningioma | pituitary | notumor`; `tumor_detected` derived
- localization: bbox `xyxy` original pixels, centroid of largest, area=sum, empty=`None/None/0`
- calibration: T + τ1/τ2 on validation only, then LOCK
- split: test 1000 LOCKED; pool 5000 → 4000/1000, stratified, seed 42
- one `InferenceService` (`src/brain_tumor/inference/service.py`); Streamlit direct-call

## Run Data Gate 0
```powershell
python scripts/data_gate/run_gate.py
```
Exit 0=PASS, 2=FAIL. Evidence → `outputs/data_gate_0/`.
Place dataset under `data/` first; Gate FAILs honestly when absent.

## Layout
`configs/ docs/ src/brain_tumor/ app/api app/streamlit scripts/data_gate tests/`
