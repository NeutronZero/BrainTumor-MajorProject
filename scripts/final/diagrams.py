"""Generate frozen-design architecture diagrams (matplotlib, no model/data).

Outputs docs/architecture/system_flow.png (inference pipeline) and
docs/architecture/release_chain.png (evidence/release progression).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _boxes(ax, items, x0=0.05, width=0.9):
    n = len(items)
    h = 0.86 / n
    for i, (title, sub, color) in enumerate(items):
        y = 0.93 - (i + 1) * h + 0.02
        ax.add_patch(FancyBboxPatch((x0, y), width, h - 0.03,
                                    boxstyle="round,pad=0.01", fc=color, ec="black"))
        ax.text(0.5, y + (h - 0.03) / 2 + 0.015, title, ha="center", fontsize=10,
                weight="bold")
        if sub:
            ax.text(0.5, y + (h - 0.03) / 2 - 0.02, sub, ha="center", fontsize=8)
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((0.5, y - 0.005), (0.5, y - h + 0.035),
                                         mutation_scale=12, color="black"))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    out = root / "docs" / "architecture"
    out.mkdir(parents=True, exist_ok=True)

    model, obs, frozen = "#cfe2ff", "#fff3cd", "#d1e7dd"
    fig, ax = plt.subplots(figsize=(8, 10))
    _boxes(ax, [
        ("Input bytes", "upload + 10MB limit", "white"),
        ("Quality gate Q01-Q09", "descriptive; never blocks inference", obs),
        ("Classification CLS-001", "ConvNeXt-Tiny, T=0.5116, τ1/τ2 frozen", model),
        ("Segmentation SEG-001", "U-Net, thr 0.5 (tumor preds only)", model),
        ("Localization", "bbox/centroid/area, original px", model),
        ("Consistency UNC-001", "K=8 σ=0.05, descriptive", obs),
        ("Reliability REL-001", "stable/review + basis, descriptive", obs),
        ("System state", "frozen precedence", frozen),
        ("API / Streamlit", "contract-validated outputs", frozen),
    ])
    fig.suptitle("SB-1.x inference pipeline (frozen)")
    fig.savefig(out / "system_flow.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    _boxes(ax, [
        ("SB-1 → SB-1.1", "baselines + packaging", frozen),
        ("ADVANCED-FOUNDATION → ADVANCED-R1", "capabilities + portability", frozen),
        ("VV-001 → ENG-001 → OFF-001", "verification + requirements", obs),
        ("CORRECTION-001 → ENG-002", "audit fixes + productization", model),
    ])
    fig.suptitle("Release / evidence chain (immutable tags)")
    fig.savefig(out / "release_chain.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("diagrams written to docs/architecture/")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
