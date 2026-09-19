"""Validation-only calibration: temperature NLL + tau selection, then LOCK."""

from __future__ import annotations

import torch


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, init: float = 1.0) -> float:
    """LBFGS on validation NLL. Fail-safe: any optimizer failure returns T=1.0
    with a flag-free honest value (caller records temperature; T=1.0 = no-op)."""
    import torch.nn as nn
    logits, labels = logits.detach().cpu(), labels.detach().cpu()
    T = torch.nn.Parameter(torch.tensor([init]))
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50)

    def _loss():
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(logits / T.clamp(min=1e-3), labels)
        loss.backward()
        return loss

    try:
        opt.step(_loss)
        T_val = float(T.detach().clamp(min=1e-3))
        if not (0.05 < T_val < 20.0):
            raise ValueError(f"implausible temperature {T_val}")
        return T_val
    except Exception as e:  # noqa: BLE001 — calibration must never abort a run
        print(f"fit_temperature: LBFGS failed ({e}); falling back to T=1.0")
        return 1.0


def ece(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    conf, pred = probs.max(dim=1)
    edges = torch.linspace(0, 1, bins + 1)
    out = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.any():
            out += (m.float().mean() * abs(pred[m].eq(labels[m]).float().mean() - conf[m].mean())).item()
    return out


def select_taus(probs: torch.Tensor, labels: torch.Tensor):
    """Grid-search tau1 (max prob) x tau2 (margin) maximizing
    val_macro_f1 - 0.5 * uncertain_rate. Validation only. Criterion recorded
    in run metrics; thresholds locked before any test evaluation."""
    from sklearn.metrics import f1_score
    top2 = probs.topk(2, dim=1).values
    conf, margin = top2[:, 0], top2[:, 0] - top2[:, 1]
    pred = probs.argmax(dim=1).numpy()
    y = labels.numpy()
    best = (-1.0, 0.80, 0.20)
    t1 = [round(x, 2) for x in np_arange(0.5, 0.96, 0.05)]
    t2 = [round(x, 2) for x in np_arange(0.05, 0.41, 0.05)]
    for a in t1:
        for b in t2:
            certain = (conf.numpy() >= a) & (margin.numpy() >= b)
            score_f1 = f1_score(y[certain], pred[certain], average="macro", zero_division=0) if certain.any() else 0.0
            score = score_f1 - 0.5 * (1 - certain.mean())
            if score > best[0]:
                best = (score, a, b)
    return {"criterion": "val_macro_f1_minus_half_uncertain_rate",
            "tau1": best[1], "tau2": best[2], "score": best[0]}


def np_arange(a: float, b: float, s: float):
    out, x = [], a
    while x < b + 1e-9:
        out.append(x)
        x += s
    return out
