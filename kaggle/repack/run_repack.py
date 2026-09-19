"""Repack SEG-001 best.pt into state-only artifact (large-file retrieval).

The full checkpoint (~370 MB with optimizer/scheduler/RNG) repeatedly breaks
this leg's connection mid-transfer. The registry's production need is the
exact fp32 state_dict, so this kernel re-saves state-only (~124 MB) plus a
SHA256 sidecar. No training, no evaluation, no science — pure repackaging of
our own COMPLETE kernel output. Verifies val_metric + finiteness before save.
"""

import hashlib
import sys
from pathlib import Path

WORK = Path("/kaggle/working")


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


src = find_under(Path("/kaggle/input"), "best.pt")
assert src is not None, "best.pt not found under /kaggle/input"
print(f"src={src} bytes={src.stat().st_size}", flush=True)
assert src.stat().st_size > 100_000_000, "best.pt suspiciously small"

import torch

ckpt = torch.load(src, map_location="cpu", weights_only=False)  # own file
print(f"keys={sorted(ckpt.keys())} epoch={ckpt.get('epoch')} "
      f"val={ckpt.get('val_metric')}", flush=True)
sd = ckpt["state"]
bad = [k for k, v in sd.items() if not torch.isfinite(v).all()]
assert not bad, f"non-finite tensors: {bad}"

out = WORK / "best_state.pt"
torch.save({"state": sd, "epoch": ckpt.get("epoch"),
            "val_metric": ckpt.get("val_metric"),
            "source": "SEG-001 best.pt repack, state-only, exact fp32"}, out)
h = hashlib.sha256(out.read_bytes()).hexdigest()
(WORK / "best_state.sha256").write_text(h + "  best_state.pt\n")
print(f"saved {out.stat().st_size} sha256={h}", flush=True)
print("REPACK DONE", flush=True)
