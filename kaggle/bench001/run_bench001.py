"""Kaggle T4 bootstrap for the FP32 GPU latency reference (measurement only).

No training, no data dependency (synthetic inputs), no model changes.
Entry point: scripts/bench/gpu_latency.py.
Output: outputs/SYSINT/gpu_latency_fp32.json.
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
code_in = find_under(Path("/kaggle/input"), "gpu_latency.py")
assert code_in is not None, "code snapshot (gpu_latency.py) not found under /kaggle/input"
code_root = code_in.parents[2]  # .../scripts/bench/<file> -> snapshot root
print(f"code_root={code_root}", flush=True)
for need in ("checkpoints/CLS-001/best.pt", "checkpoints/SEG-001/best.pt",
             "outputs/CLS-001/calibration_frozen.json", "outputs/SEG-001/metrics.json"):
    assert (code_root / need).is_file(), f"snapshot missing {need} — aborting"
if PROJ.exists():
    shutil.rmtree(PROJ)
shutil.copytree(code_root, PROJ, symlinks=True)
os.chdir(PROJ)

# 2. Deps (image ships CUDA torch; CPU-safe numpy/PIL/sklearn for completeness).
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pyyaml", "scikit-learn", "scipy", "pandas", "pillow"])
import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)
assert torch.cuda.is_available(), "no CUDA on a GPU kernel — aborting"

# 3. Measurement.
r = subprocess.run([sys.executable, "scripts/bench/gpu_latency.py"])
assert r.returncode == 0, "gpu_latency failed"
print("GPU LATENCY KERNEL DONE", flush=True)
