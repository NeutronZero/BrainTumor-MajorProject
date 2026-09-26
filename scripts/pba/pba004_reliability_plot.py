"""PBA-004 reliability diagram — descriptive plot from FROZEN artifacts only.

Reads (no model inference, no recalibration, no threshold changes):
  outputs/PBA-001/per_case_test.json   (frozen CLS-001 test probs)
  outputs/PBA-004/pba004.json          (binned calibration data)
  outputs/test_evaluation_7b860dca72ea.json (locked eval artifact for verification)
  outputs/CLS-001/calibration_frozen.json   (frozen T/tau1/tau2, display only)

Writes:
  outputs/PBA-004/reliability_diagram.png
  outputs/PBA-004/reliability_data.json  (plot source data + verification)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
BINS = 15


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out4 = root / "outputs" / "PBA-004"

    test = json.loads((root / "outputs" / "PBA-001" / "per_case_test.json").read_text())
    rep4 = json.loads((out4 / "pba004.json").read_text())
    cal = json.loads((root / "outputs" / "CLS-001" / "calibration_frozen.json").read_text())
    ev = json.loads((root / "outputs" / "test_evaluation_7b860dca72ea.json").read_text())

    # Recompute bins directly from frozen per-case probs (verification, not refit)
    P = np.array([c["probs"] for c in test])
    Y = np.array([CLASSES.index(c["true"]) for c in test])
    pred = P.argmax(1)
    conf = P.max(1)
    correct = (pred == Y).astype(float)

    plot_bins = []
    ece_check = 0.0
    for i in range(BINS):
        lo, hi = i / BINS, (i + 1) / BINS
        m = (conf > lo) & (conf <= hi)
        n = int(m.sum())
        if n:
            acc, avg = float(correct[m].mean()), float(conf[m].mean())
            ece_check += (n / len(test)) * abs(acc - avg)
            plot_bins.append(
                {
                    "bin": f"{lo:.2f}-{hi:.2f}",
                    "lo": lo,
                    "hi": hi,
                    "center": (lo + hi) / 2,
                    "n": n,
                    "accuracy": acc,
                    "mean_confidence": avg,
                    "gap": acc - avg,
                }
            )
        else:
            plot_bins.append(
                {
                    "bin": f"{lo:.2f}-{hi:.2f}",
                    "lo": lo,
                    "hi": hi,
                    "center": (lo + hi) / 2,
                    "n": 0,
                    "accuracy": None,
                    "mean_confidence": None,
                    "gap": None,
                }
            )

    ev_ece = ev["cls_primary"]["ece_post_T"]
    stored_ece = rep4["ece_post_T"]
    assert abs(ece_check - ev_ece) < 1e-6, f"recomputed {ece_check} vs locked-eval {ev_ece}"
    assert abs(stored_ece - ev_ece) < 1e-6, f"stored {stored_ece} vs locked-eval {ev_ece}"
    assert ev["cls_primary"]["uncertain"] == len(rep4["uncertain_cases"]) == 4
    assert len(rep4["confident_but_wrong"]) == 2, "CBW must be 2 (certain errors only)"
    assert len(test) == ev["cls_primary"]["n"] == 1000

    nonempty = [b for b in plot_bins if b["n"] > 0]
    centers = np.array([b["mean_confidence"] for b in nonempty])
    accs = np.array([b["accuracy"] for b in nonempty])
    ns = np.array([b["n"] for b in nonempty])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    ax1.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="perfect calibration")
    ax1.scatter(centers, accs, s=np.clip(ns, 10, 400), alpha=0.8, label="observed (size ~ n)")
    for b in nonempty:
        ax1.text(
            b["mean_confidence"], b["accuracy"], f" n={b['n']}", fontsize=7, va="bottom", ha="left"
        )
    ax1.set_ylabel("accuracy")
    ax1.set_title(
        f"CLS-001 reliability (frozen T={cal['temperature']:.4f}, N=1000, "
        f"ECE={ev_ece:.4f}, CBW=2, uncertain=4)"
    )
    ax1.set_xlim(0.5, 1.0)
    ax1.set_ylim(0.0, 1.02)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    all_centers = np.array([b["center"] for b in plot_bins])
    all_n = np.array([b["n"] for b in plot_bins])
    ax2.bar(all_centers, all_n, width=1 / BINS * 0.9)
    ax2.set_xlabel("confidence")
    ax2.set_ylabel("count")
    ax2.set_xlim(0.5, 1.0)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    png = out4 / "reliability_diagram.png"
    fig.savefig(png, dpi=150)
    plt.close(fig)

    data = {
        "source": "frozen per_case_test.json probs; descriptive only, no refit",
        "frozen": {"T": cal["temperature"], "tau1": cal["tau1"], "tau2": cal["tau2"]},
        "N": len(test),
        "ece_recomputed": round(float(ece_check), 6),
        "ece_locked_eval": ev_ece,
        "ece_pba004_stored": stored_ece,
        "verification": "recomputed == locked-eval == stored (tol 1e-6); N=1000; uncertain=4; CBW=2",
        "confident_but_wrong_n": 2,
        "uncertain_n": 4,
        "bins_15": plot_bins,
        "plot": str(png.name),
    }
    (out4 / "reliability_data.json").write_text(json.dumps(data, indent=1))
    print(
        json.dumps(
            {
                "png": str(png),
                "ece": ev_ece,
                "bins_nonempty": len(nonempty),
                "cbw": 2,
                "uncertain": 4,
                "verified": True,
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
