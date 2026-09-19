"""Kaggle T4 bootstrap for the FP16 rung (measurement only, autocast precision).

Needs the BRISC data mount (validation populations live there) + code snapshot.
Entry point: scripts/bench/fp16_check.py. No training, no model changes.
Output: outputs/SYSINT/fp16_check.json.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
PROJ = WORK / "project"
print(f"code={Path(__file__).parent} work={WORK} python={sys.version.split()[0]}", flush=True)


def find_under(base: Path, target: str, max_depth: int = 8):
    cands = [base] if base.is_dir() else []
    for _ in range(max_depth):
        nxt = []
        for d in cands:
            try:
                kids = list(d.iterdir())
            except OSError:
                continue
            for k in kids:
                if k.name == target:
                    return k
                if k.is_dir() and not k.is_symlink():
                    nxt.append(k)
        cands = nxt
    return None


# 1. Stage writable code snapshot.
code_in = find_under(Path("/kaggle/input"), "fp16_check.py")
assert code_in is not None, "code snapshot (fp16_check.py) not found under /kaggle/input"
code_root = code_in.parents[2]  # .../scripts/bench/<file> -> snapshot root
print(f"code_root={code_root}", flush=True)
for need in ("checkpoints/CLS-001/best.pt", "checkpoints/SEG-001/best.pt",
             "outputs/CLS-001/calibration_frozen.json", "outputs/SEG-001/metrics.json",
             "outputs/data_gate_0/project_manifest.csv"):
    assert (code_root / need).is_file(), f"snapshot missing {need} — aborting"
if PROJ.exists():
    shutil.rmtree(PROJ)
shutil.copytree(code_root, PROJ, symlinks=True)
os.chdir(PROJ)

# 2. Data link for validation populations (robust to input mount layout).
explicit = [Path("/kaggle/input/datasets/briscdataset/brisc2025"),
            Path("/kaggle/input/brisc2025")]
cls_task = None
for cand in explicit:
    probe = cand / "classification_task"
    if cand.is_dir() and probe.is_dir():
        cls_task = probe
        print(f"explicit hit: {cand}", flush=True)
        break
    inner = cand / "brisc2025"
    if inner.is_dir() and (inner / "classification_task").is_dir():
        cls_task = inner / "classification_task"
        print(f"explicit hit (nested): {inner}", flush=True)
        break
if cls_task is None:
    cls_task = find_under(Path("/kaggle/input"), "classification_task")
assert cls_task is not None, "classification_task not found under /kaggle/input"
release = cls_task.parent
link = PROJ / "data" / "brisc2025"
link.parent.mkdir(parents=True, exist_ok=True)
if link.is_symlink() or link.exists():
    link.unlink() if link.is_symlink() else shutil.rmtree(link)
try:
    link.symlink_to(release, target_is_directory=True)
except OSError:
    shutil.copytree(release, link, symlinks=True)
n_tr = sum(1 for _ in (link / "classification_task" / "train").rglob("*.jpg"))
n_te = sum(1 for _ in (link / "classification_task" / "test").rglob("*.jpg"))
print(f"data train={n_tr} test={n_te}", flush=True)
assert (n_tr, n_te) == (5000, 1000), "data drift — aborting, not adapting"

# 3. Deps.
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pyyaml", "scikit-learn", "scipy", "pandas", "pillow"])
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)
assert torch.cuda.is_available(), "no CUDA on a GPU kernel — aborting"

# 4. Measurement.
r = subprocess.run([sys.executable, "scripts/bench/fp16_check.py"])
assert r.returncode == 0, "fp16_check failed"
print("FP16 KERNEL DONE", flush=True)
