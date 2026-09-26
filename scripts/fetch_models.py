"""fetch_models.py — verify pinned checkpoints are present (§16).

Checkpoints ride via Git LFS (see .gitattributes); a fresh clone needs
`git lfs pull` or the *.pt files are 1-KB pointer text and inference will
report unavailable. No network step beyond LFS — nothing to download here.
"""

from __future__ import annotations

import sys
from pathlib import Path

EXPECTED = [
    "checkpoints/CLS-001/best.pt",
    "checkpoints/SEG-001/best.pt",
    "checkpoints/ROB-001/best.pt",
]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    missing = [p for p in EXPECTED if not (root / p).exists()]
    tiny = [
        p
        for p in EXPECTED
        if (root / p).exists() and (root / p).stat().st_size < 1024 * 1024
    ]
    if not missing and not tiny:
        print("fetch_models: OK — all pinned checkpoints present.")
        return 0
    for p in missing:
        print(f"fetch_models: MISSING {p}")
    for p in tiny:
        print(f"fetch_models: LFS pointer? {p} is <1MB — run `git lfs pull`.")
    print("Without checkpoints the API/UI start but inference is unavailable.")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
