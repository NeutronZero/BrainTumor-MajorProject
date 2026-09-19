"""train_rob001.py — ROB-001. Kaggle T4 run default; --smoke for CPU pipeline check.

Mirrors scripts/train/train_segmenter.py exactly except:
  - exp_id ROB-001 (checkpoints/ROB-001/, outputs/ROB-001/); SEG-001 files untouched.
  - train loader passes augmentation.gauss_noise_sigma from config (train-only);
    val loader is noise-free and deterministic.
Validation Dice selection; test LOCKED (not evaluated here).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.data.brisc import SegDataset, load_manifest  # noqa: E402
from brain_tumor.preprocessing.pipeline import build_seg_pair_transform  # noqa: E402
from brain_tumor.segmentation.unet import UNet, dice_bce_loss  # noqa: E402
from brain_tumor.utils.engine import fit  # noqa: E402
from brain_tumor.utils.seed import make_generator, set_global_seed, worker_init_fn  # noqa: E402
from brain_tumor.utils.amp import require_cuda_arch  # noqa: E402

EXP_ID = "ROB-001"


def _seeded_worker(base_seed: int):
    def _fn(worker_id: int):
        worker_init_fn(worker_id, base_seed)
    return _fn


def seg_stats(project_root: Path, manifest, n_sample: int = 200):
    from PIL import Image
    import numpy as np
    ds = SegDataset(project_root, manifest, "train")
    idx = range(0, min(n_sample, len(ds)), max(1, min(n_sample, len(ds)) // 50))
    px, skipped = [], 0
    for i in idx:
        try:
            a = np.asarray(Image.open(ds.pairs[i][0]).convert("L"), dtype=np.float64) / 255.0
        except Exception:
            skipped += 1
            continue
        px.append(a.ravel()[::97])
    if not px:
        raise RuntimeError("seg_stats: no readable images")
    if skipped:
        print(f"seg_stats: skipped {skipped} unreadable images")
    px = np.concatenate(px)
    return float(px.mean()), float(px.std() + 1e-8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/ROB-001.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(root / args.config))
    set_global_seed(cfg.get("seed", 42))
    device = require_cuda_arch()
    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")

    mean, std = (0.5, 0.25) if args.smoke else seg_stats(root, manifest)
    aug = cfg.get("augmentation", {})
    kw = dict(image_size=cfg.get("image_size", 256), mean=mean, std=std)
    kw_train = dict(kw, gauss_noise_sigma=float(aug.get("gauss_noise_sigma", 0.0)))

    def _loader(split: str, train: bool, bs: int):
        ds = SegDataset(root, manifest, split,
                        pair_transform=build_seg_pair_transform(
                            train, **(kw_train if train else kw)))
        if args.subset or args.smoke:
            ds.pairs = ds.pairs[:args.subset or 32]
        return torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=train, num_workers=0,
            worker_init_fn=_seeded_worker(cfg.get("seed", 42)),
            generator=make_generator(cfg.get("seed", 42) + 3),
            pin_memory=torch.cuda.is_available())

    bs = 4 if args.smoke else cfg.get("batch_size", 16)
    epochs = args.epochs or (1 if args.smoke else cfg.get("epochs_max", 30))
    tr, va = _loader("train", True, bs), _loader("val", False, bs)

    model = UNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"],
                            weight_decay=cfg["optim"].get("weight_decay", 0.01))
    sched = _warmup_cosine(opt, epochs, cfg["optim"].get("warmup_epochs", 3))
    loss_cfg = cfg.get("loss", {})
    loss_fn = lambda o, t: dice_bce_loss(o, t, loss_cfg.get("dice_weight", 1.0),
                                         loss_cfg.get("bce_weight", 1.0))

    def _val_metric(m, loader, dev):
        m.eval()
        inter = union = 0.0
        with torch.no_grad():
            for xb, yb in loader:
                p = (torch.sigmoid(m(xb.to(dev))) > 0.5).float()
                yb = yb.to(dev)
                inter += (p * yb).sum().item()
                union += p.sum().item() + yb.sum().item()
        return float((2 * inter + 1e-6) / (union + 1e-6))

    ckpt = root / "checkpoints" / (EXP_ID + "-smoke" if args.smoke else EXP_ID) / "best.pt"
    es = cfg.get("early_stopping", {})
    res = fit(model, tr, va, loss_fn, opt, sched, device, epochs,
              es.get("patience", 8), es.get("min_delta", 0.001), ckpt,
              amp=cfg.get("stability", {}).get("amp", True),
              clip_norm=cfg.get("stability", {}).get("grad_clip_norm", 1.0),
              val_metric_fn=_val_metric)
    import hashlib
    cfg_bytes = open(root / args.config, "rb").read()
    hist = res.get("history", [])
    best_epoch = max(range(len(hist)), key=lambda i: hist[i]["val_metric"]) if hist else None
    metrics = {"val_dice": res["best_val_metric"], "epochs_run": res["epochs_run"],
               "epochs_max": epochs, "best_epoch": best_epoch,
               "stopping_reason": ("early_stopped" if res["epochs_run"] < epochs
                                   else "completed_all_epochs"),
               "val_dice_per_epoch": hist,
               "norm": {"mean": mean, "std": std}, "device": device, "smoke": args.smoke,
               "train_noise_sigma": kw_train["gauss_noise_sigma"],
               "config_sha256": hashlib.sha256(cfg_bytes).hexdigest()}

    exp_id = EXP_ID + ("-smoke" if args.smoke else "")
    outd = root / "outputs" / exp_id
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (outd / "run_config.json").write_text(json.dumps(
        {"config": args.config, "seed": cfg.get("seed"), "device": device,
         "torch": torch.__version__, "git": _git(root),
         "time": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=1))
    (outd / "config.yaml").write_text(open(root / args.config).read())
    (outd / "git_commit.txt").write_text(_git(root))
    (outd / "notes.md").write_text(f"# {exp_id}\nH-ROBUST-001 run. Test LOCKED (not evaluated).\n")
    print(json.dumps(metrics, indent=1))
    return 0


def _warmup_cosine(opt, epochs: int, warmup: int):
    import torch as _t
    warm = max(0, min(warmup, epochs - 1)) if epochs > 1 else 0
    if warm <= 0:
        return _t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    w = _t.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=warm)
    c = _t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warm))
    return _t.optim.lr_scheduler.SequentialLR(opt, [w, c], milestones=[warm])


def _git(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
