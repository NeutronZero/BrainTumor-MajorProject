"""fetch_models.py — downloads pinned checkpoints + verifies SHA256 (§16). Only net step."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    print("fetch_models: no remote release pinned yet (Phase 0). Nothing to download.")
    print("See models/README.md for name/experiment/release/SHA256/location.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
