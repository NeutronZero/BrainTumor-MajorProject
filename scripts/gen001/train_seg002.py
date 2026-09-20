"""train_seg002.py — GEN-001/SEG-002. Clone of train_segmenter.py (SEG-001).

ONLY deliberate deltas vs SEG-001 trainer:
  1. model class: AttentionUNet (import) instead of UNet;
  2. --seed {42,43,44} (required; SEG-001 used config seed 42);
  3. outputs -> outputs/GEN-001/seed<SEED>/ ; checkpoints -> checkpoints/GEN-001/seed<SEED>/;
  4. norm drift-guard: recomputed seg-train stats must match SEG-001 recorded
     (mean 0.10167703278129603, std 0.14063970563559394) within 1e-12, else abort.

Everything else — data rules, split, loss, optim, schedule, batch, epochs,
early stopping, AMP/clip, augmentation, metric, engine.fit — is byte-identical
logic. Test is never touched (SegDataset globs segmentation_task/train only).
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
from brain_tumor.segmentation.attention_unet import AttentionUNet, count_parameters  # noqa: E402
from brain_tumor.segmentation.unet import dice_bce_loss  # noqa: E402 (shared loss fn)
from brain_tumor.utils.engine import fit  # noqa: E402
from brain_tumor.utils.seed import make_generator, set_global_seed, worker_init_fn  # noqa: E402
from brain_tumor.utils.amp import require_cuda_arch  # noqa: E402

SEG001_NORM = {"mean": 0.10167703278129603, "std": 0.14063970563559394}


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
    ap.add_argument("--config", default="configs/experiment/GEN-001.yaml")
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43, 44])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(root / args.config))
    assert args.seed in cfg.get("seeds", [42, 43, 44]), "seed not in frozen GEN-001 set"
    set_global_seed(args.seed)
    device = require_cuda_arch()
    manifest = load_manifest(root / "outputs" / "data_gate_0" / "project_manifest.csv")

    mean, std = (0.5, 0.25) if args.smoke else seg_stats(root, manifest)
    if not args.smoke:
        assert abs(mean - SEG001_NORM["mean"]) < 1e-12, f"norm mean drift: {mean}"
        assert abs(std - SEG001_NORM["std"]) < 1e-12, f"norm std drift: {std}"
        print(f"norm drift-guard PASS: mean={mean} std={std}", flush=True)
    kw = dict(image_size=cfg.get("image_size", 256), mean=mean, std=std)

    def _loader(split: str, train: bool, bs: int):
        ds = SegDataset(root, manifest, split,
                        pair_transform=build_seg_pair_transform(train, **kw))
        if args.subset or args.smoke:
            ds.pairs = ds.pairs[:args.subset or 32]
        return torch.utils.data.DataLoader(
            ds, batch_size=bs, shuffle=train, num_workers=0,
            worker_init_fn=_seeded_worker(args.seed),
            generator=make_generator(args.seed + 3),
            pin_memory=torch.cuda.is_available())

    bs = 4 if args.smoke else cfg.get("batch_size", 16)
    epochs = args.epochs or (1 if args.smoke else cfg.get("epochs_max", 30))
    tr, va = _loader("train", True, bs), _loader("val", False, bs)
    print(f"GEN-001 seed={args.seed}: train pairs={len(tr.dataset)} "
          f"val pairs={len(va.dataset)}", flush=True)

    model = AttentionUNet().to(device)
    n_params = count_parameters(model)
    print(f"SEG-002 params={n_params}", flush=True)
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

    tag = f"seed{args.seed}" + ("-smoke" if args.smoke else "")
    ckpt = root / "checkpoints" / "GEN-001" / tag / "best.pt"
    es = cfg.get("early_stopping", {})
    res = fit(model, tr, va, loss_fn, opt, sched, device, epochs,
              es.get("patience", 8), es.get("min_delta", 0.001), ckpt,
              amp=cfg.get("stability", {}).get("amp", True),
              clip_norm=cfg.get("stability", {}).get("grad_clip_norm", 1.0),
              val_metric_fn=_val_metric)
    metrics = {"experiment": "GEN-001", "seed": args.seed, "params": n_params,
               "val_dice_global": res["best_val_metric"], "epochs_run": res["epochs_run"],
               "history": res["history"],
               "norm": {"mean": mean, "std": std}, "device": device, "smoke": args.smoke}

    outd = root / "outputs" / "GEN-001" / tag
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (outd / "run_config.json").write_text(json.dumps(
        {"config": args.config, "seed": args.seed, "device": device,
         "torch": torch.__version__, "git": _git(root),
         "time": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=1))
    (outd / "config.yaml").write_text(open(root / args.config).read())
    (outd / "git_commit.txt").write_text(_git(root))
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=1))
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
