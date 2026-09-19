"""Kaggle T4 bootstrap for CLS-001 (full 30-epoch prescribed run).

Layout reality (v1 evidence): kernel code is read-only at /kaggle/src
(script.py only); writable area is /kaggle/working. Code snapshot arrives as
Kaggle dataset input; BRISC bytes attached as Kaggle input (never re-uploaded).
No config changes, no test evaluation.
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


def _tree(base: Path, depth: int = 3, cap: int = 40):
    out = []
    def _walk(d: Path, lvl: int):
        if lvl > depth or len(out) >= cap:
            return
        try:
            kids = sorted(d.iterdir())
        except OSError:
            return
        for k in kids[:cap]:
            out.append("  " * lvl + k.name + ("/" if k.is_dir() else ""))
            if k.is_dir() and not k.is_symlink():
                _walk(k, lvl + 1)
    _walk(base, 0)
    return "\n".join(out)


# 1. Stage writable code snapshot.
code_in = find_under(Path("/kaggle/input"), "train_classifier.py")
assert code_in is not None, "code snapshot (train_classifier.py) not found under /kaggle/input"
code_root = code_in.parents[2]  # .../scripts/train/<file> -> snapshot root
print(f"code_root={code_root}", flush=True)
if PROJ.exists():
    shutil.rmtree(PROJ)
shutil.copytree(code_root, PROJ, symlinks=True)
os.chdir(PROJ)

# 2. Data link (robust to input mount layout).
print("input tree:\n" + _tree(Path("/kaggle/input")), flush=True)
din = Path("/kaggle/input/datasets")
if din.is_dir():
    print("datasets/: " + str(sorted(p.name for p in din.iterdir())[:20]), flush=True)
    for owner in sorted(din.iterdir()):
        if owner.is_dir():
            print(f"datasets/{owner.name}/: " + str(sorted(p.name for p in owner.iterdir())[:20]), flush=True)
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

# 3. Deps (image already ships CUDA torch/torchvision).
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pydantic", "pyyaml", "scikit-learn", "scipy", "pandas"])
import torch

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} "
      f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}", flush=True)
assert torch.cuda.is_available(), "no CUDA on a GPU kernel — aborting"

# 4. Full prescribed run (validation-only calibration inside; test locked).
r = subprocess.run([sys.executable, "scripts/train/train_classifier.py"])
assert r.returncode == 0, "train_classifier failed"
print("CLS-001 KERNEL DONE", flush=True)
