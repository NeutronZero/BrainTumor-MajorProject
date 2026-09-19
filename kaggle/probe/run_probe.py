"""GPU AMP diagnostic probe (not an experiment): 20 batches, fp32 vs fp16.

Stages code + data exactly like the CLS-001 kernel, then reports per-batch
loss / grad-norm / scaler-scale and the first non-finite gradient location.
No checkpoints, no config changes, no test access.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
print(f"code={Path(__file__).parent} python={sys.version.split()[0]}", flush=True)


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


code_in = find_under(Path("/kaggle/input"), "train_classifier.py")
assert code_in is not None, "code snapshot not found"
PROJ = WORK / "project"
if PROJ.exists():
    shutil.rmtree(PROJ)
shutil.copytree(code_in.parents[2], PROJ, symlinks=True)
os.chdir(PROJ)
sys.path.insert(0, str(PROJ / "src"))

cls_task = find_under(Path("/kaggle/input"), "classification_task")
release = cls_task.parent
link = PROJ / "data" / "brisc2025"
link.parent.mkdir(parents=True, exist_ok=True)
if link.is_symlink() or link.exists():
    link.unlink() if link.is_symlink() else shutil.rmtree(link)
try:
    link.symlink_to(release, target_is_directory=True)
except OSError:
    shutil.copytree(release, link, symlinks=True)

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       "pydantic", "pyyaml", "scikit-learn", "scipy", "pandas"])
import torch
import yaml

assert torch.cuda.is_available()
print(f"torch={torch.__version__} {torch.cuda.get_device_name(0)}", flush=True)

from brain_tumor.classification.models import build_classifier
from brain_tumor.data.brisc import ClsDataset, load_manifest
from brain_tumor.preprocessing.pipeline import build_cls_transform
from brain_tumor.utils.seed import set_global_seed

set_global_seed(42)
root = PROJ
manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")
ds = ClsDataset(root, manifest, "train", transform=build_cls_transform(True))
import numpy as np
rng = np.random.RandomState(0)
ds.rows = [ds.rows[i] for i in rng.choice(len(ds.rows), 320, replace=False)]
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
print(f"probe batches: {len(loader)}", flush=True)


def first_bad(model):
    for n, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            return n, str(p.grad.abs().max().item())
    return None, None


def run(tag, use_amp):
    import torch.nn as nn
    torch.manual_seed(0)
    model = build_classifier("convnext_tiny", pretrained=True).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    for b, (xb, yb) in enumerate(loader):
        if b >= 5:
            break
        xb, yb = xb.cuda(), yb.cuda()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = loss_fn(model(xb), yb)
        lf = float(loss.detach())
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
        gnf = float(gn)
        n_bad, mx = first_bad(model)
        print(f"{tag} batch={b} loss={lf:.4f} grad_norm={gnf:.4e} "
              f"scale={float(scaler.get_scale()):.2e} first_bad={n_bad} {mx}", flush=True)
        if use_amp:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()


run("fp32", False)
run("fp16", True)
print("PROBE DONE", flush=True)
