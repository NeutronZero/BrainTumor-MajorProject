"""train_classifier.py — CLS-001. Full T4 run default; --smoke for CPU pipeline check."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # silence torchrun-style OMP blocks (prior A4)

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.classification import calibrate, models  # noqa: E402
from brain_tumor.config import sha256_dict  # noqa: E402
from brain_tumor.data.brisc import (  # noqa: E402
    CLASS_TO_IDX,
    ClsDataset,
    load_manifest,
    stratified_sampler_weights,
)
from brain_tumor.preprocessing.pipeline import build_cls_transform  # noqa: E402
from brain_tumor.utils.engine import fit  # noqa: E402
from brain_tumor.utils.seed import make_generator, set_global_seed, worker_init_fn  # noqa: E402
from brain_tumor.utils.amp import require_cuda_arch  # noqa: E402


def _seeded_worker(base_seed: int):
    def _fn(worker_id: int):
        worker_init_fn(worker_id, base_seed)
    return _fn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment/CLS-001.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--subset", type=int, default=None)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(root / args.config))
    set_global_seed(cfg.get("seed", 42))
    device = require_cuda_arch()  # fail-fast on sm<70 (prior F4); cpu if no CUDA

    manifest = load_manifest(root / cfg["data"]["manifest"])
    _assert_pool_counts(manifest)  # fail-closed (prior B7 lesson)
    seed = cfg.get("seed", 42)

    def _loader(split: str, train: bool, bs: int, subset=None):
        ds = ClsDataset(root, manifest, split,
                        transform=build_cls_transform(train, cfg.get("image_size", 224)))
        if subset:  # stratified interleave so smoke subsets span all classes
            by_cls: dict = {}
            for r in ds.rows:
                by_cls.setdefault(r["class"], []).append(r)
            inter, i = [], 0
            while len(inter) < subset and any(v[i:] for v in by_cls.values()):
                for c in sorted(by_cls):
                    if i < len(by_cls[c]):
                        inter.append(by_cls[c][i])
                        if len(inter) >= subset:
                            break
                i += 1
            ds.rows = inter
        labels = [CLASS_TO_IDX[r["class"]] for r in ds.rows]
        # Dedicated generators: sampler and loader must not share one (prior audit).
        sampler = stratified_sampler_weights(
            labels, generator=make_generator(seed + 1)) if train else None
        return torch.utils.data.DataLoader(
            ds, batch_size=bs, sampler=sampler, shuffle=False,
            num_workers=0, worker_init_fn=_seeded_worker(seed),
            generator=make_generator(seed + 2),
            pin_memory=torch.cuda.is_available())

    bs = 8 if args.smoke else cfg.get("batch_size", 64)
    epochs = args.epochs or (1 if args.smoke else cfg.get("epochs_max", 30))
    tr = _loader("train", True, bs, args.subset or (64 if args.smoke else None))
    va = _loader("val", False, bs, args.subset // 2 if args.subset else (32 if args.smoke else None))

    model = models.build_classifier(cfg.get("architecture", "convnext_tiny"),
                                    pretrained=(not args.smoke) and cfg.get("init", "").startswith("imagenet"))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["optim"]["lr"],
                            weight_decay=cfg["optim"].get("weight_decay", 0.05))
    sched = _warmup_cosine(opt, epochs, cfg["optim"].get("warmup_epochs", 3))
    import torch.nn as nn
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg["loss"].get("eps", 0.1))

    def _val_metric(m, loader, dev):
        m.eval()
        all_p, all_l = [], []
        with torch.no_grad():
            for xb, yb in loader:
                all_p.append(m(xb.to(dev)).argmax(1).cpu())
                all_l.append(yb)
        from sklearn.metrics import f1_score
        return float(f1_score(torch.cat(all_l), torch.cat(all_p), average="macro", zero_division=0))

    ckpt = root / "checkpoints" / ("CLS-001-smoke" if args.smoke else "CLS-001") / "best.pt"
    es = cfg.get("early_stopping", {})
    fit(model, tr, va, loss_fn, opt, sched, device, epochs,
        es.get("patience", 8), es.get("min_delta", 0.001), ckpt,
        amp=cfg.get("stability", {}).get("amp", True),
        clip_norm=cfg.get("stability", {}).get("grad_clip_norm", 1.0),
        val_metric_fn=_val_metric)

    # validation-only calibration (temperature + taus), then record
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for xb, yb in va:
            logits.append(model(xb.to(device)).cpu())
            labels.append(yb)
    logits, labels = torch.cat(logits), torch.cat(labels)
    T = calibrate.fit_temperature(logits, labels)
    probs = torch.softmax(logits / T, dim=1)
    taus = calibrate.select_taus(probs, labels)
    metrics = {"val_macro_f1": _val_metric(model, va, device),
               "temperature": T, "ece_post": calibrate.ece(probs, labels),
               "taus": taus, "device": device,
               "smoke": args.smoke, "config_hash": sha256_dict(cfg)}

    exp_id = "CLS-001-smoke" if args.smoke else "CLS-001"
    outd = root / "outputs" / exp_id
    outd.mkdir(parents=True, exist_ok=True)
    (outd / "metrics.json").write_text(json.dumps(metrics, indent=1))
    (outd / "run_config.json").write_text(json.dumps(
        {"config": args.config, "seed": cfg.get("seed"), "device": device,
         "torch": torch.__version__, "git": _git(root),
         "time": datetime.datetime.now(datetime.timezone.utc).isoformat()}, indent=1))
    (outd / "config.yaml").write_text(open(root / args.config).read())
    (outd / "git_commit.txt").write_text(_git(root))
    (outd / "notes.md").write_text(f"# {exp_id}\nBaseline run. Test LOCKED (not evaluated).\n")
    _index(root, exp_id, metrics)
    print(json.dumps(metrics, indent=1))
    return 0


def _warmup_cosine(opt, epochs: int, warmup: int):
    """Linear warmup + cosine decay, exactly as frozen configs specify
    (warmup_epochs: 3). Missing warmup caused fp16 grad overflow at epoch 0
    on T4 (v5 kernel evidence) — this implements the frozen spec."""
    import torch as _t
    warm = max(0, min(warmup, epochs - 1)) if epochs > 1 else 0
    if warm <= 0:
        return _t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    w = _t.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=warm)
    c = _t.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - warm))
    return _t.optim.lr_scheduler.SequentialLR(opt, [w, c], milestones=[warm])


def _git(root: Path) -> str:
    try:
        import subprocess as _sp
        return _sp.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                text=True, stderr=_sp.DEVNULL).strip()
    except Exception:
        return "unknown"


def _index(root: Path, exp_id: str, metrics: dict):
    import csv
    p = root / "outputs" / "index.csv"
    new = not p.exists()
    # Schema matches the curated index (one row per experiment; metric
    # columns blank where unrecoverable — see outputs/CLS-001/provenance.md).
    fields = ["experiment", "metric_name", "metric_value", "device", "smoke",
              "evidence"]
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        w.writerow({"experiment": exp_id, "metric_name": "val_macro_f1",
                    "metric_value": metrics.get("val_macro_f1"),
                    "device": metrics.get("device"),
                    "smoke": metrics.get("smoke"),
                    "evidence": "metrics.json"})


def _assert_pool_counts(manifest) -> None:
    """Hard population asserts (prior B7 lesson): 4000/1000/1000(+993)."""
    pool = [r for r in manifest if r["population"] == "official_train_pool"]
    test = [r for r in manifest if r["population"] == "official_test_locked"]
    n_tr = sum(1 for r in pool if r["project_split"] == "train")
    n_va = sum(1 for r in pool if r["project_split"] == "val")
    assert (n_tr, n_va, len(test)) == (4000, 1000, 1000), \
        f"manifest drift: train={n_tr} val={n_va} test={len(test)}"


if __name__ == "__main__":
    sys.exit(main())
