"""Shared training engine: AMP, grad-clip, NaN/Inf guard, early stopping (§28/§36).

Hardened per prior audit (A9): atomic checkpoint writes (tmp + os.replace),
optimizer/scheduler/scaler/RNG states saved, grad-finiteness checked after
backward, DataParallel prefix stripped on save.
"""

from __future__ import annotations

import os
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from brain_tumor.utils.amp import autocast_if_cuda, make_grad_scaler


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _rng_states() -> dict:
    states: dict = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def _params_finite(model) -> bool:
    with torch.no_grad():
        return all(torch.isfinite(p).all().item() for p in model.parameters())


def _state_for_ckpt(model):
    """Strip DataParallel `.module` prefix on save (prior B8 lesson)."""
    import torch.nn as nn

    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def fit(
    model,
    train_loader,
    val_loader,
    loss_fn,
    optimizer,
    scheduler,
    device: str,
    epochs_max: int,
    patience: int,
    min_delta: float,
    ckpt_path: Path,
    amp: bool = True,
    clip_norm: float = 1.0,
    val_metric_fn=None,
    higher_is_better: bool = True,
):
    scaler = make_grad_scaler(amp)
    best, best_state, bad, history = None, None, 0, []
    for epoch in range(epochs_max):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_if_cuda(amp):
                out = model(xb)
                loss = loss_fn(out, yb) if yb.dtype == torch.long else loss_fn(out, yb.float())
            if not torch.isfinite(loss):
                _atomic_save(
                    {"epoch": epoch, "state": _state_for_ckpt(model)},
                    ckpt_path.parent / "failure.pt",
                )
                raise RuntimeError(f"non-finite loss at epoch {epoch}")
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            if scaler is None and not torch.isfinite(torch.as_tensor(float(grad_norm))):
                # fp32 path: no scaler to skip bad steps, so inf grads are fatal.
                _atomic_save(
                    {"epoch": epoch, "state": _state_for_ckpt(model)},
                    ckpt_path.parent / "failure.pt",
                )
                raise RuntimeError(f"non-finite grad norm at epoch {epoch}")
            # With an active GradScaler, transient fp16 overflow is routine
            # (probe: ConvNeXt stem, batch 2, recovers next batch); the scaler
            # skips the step and halves scale itself — aborting would turn
            # normal AMP semantics into a fatal error.
            if not _params_finite(model):
                _atomic_save(
                    {"epoch": epoch, "state": _state_for_ckpt(model)},
                    ckpt_path.parent / "failure.pt",
                )
                raise RuntimeError(f"non-finite params at epoch {epoch}")
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        if scheduler:
            scheduler.step()
        vm = val_metric_fn(model, val_loader, device) if val_metric_fn else 0.0
        history.append({"epoch": epoch, "val_metric": vm})
        improved = best is None or (
            vm > best + min_delta if higher_is_better else vm < best - min_delta
        )
        if improved:
            best, best_state, bad = vm, deepcopy(_state_for_ckpt(model)), 0
            _atomic_save(
                {
                    "epoch": epoch,
                    "state": best_state,
                    "val_metric": vm,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "scaler": scaler.state_dict() if scaler else None,
                    "rng": _rng_states(),
                },
                ckpt_path,
            )
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    _atomic_save(
        {
            "epoch": epoch,
            "state": _state_for_ckpt(model),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "rng": _rng_states(),
        },
        ckpt_path.parent / "last.pt",
    )
    return {"best_val_metric": best, "epochs_run": epoch + 1, "history": history}
