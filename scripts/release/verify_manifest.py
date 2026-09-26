"""Verify docs/release_manifest.md against the live tree (or a ZIP root).

Run: python scripts/release/verify_manifest.py [--root DIR]
Hashes + sizes recomputed; any mismatch or missing file exits 1.
Catches stale-manifest and CRLF/LF drift before packaging.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

LINE = re.compile(r"^- `(.+?)` `([0-9a-f]{64})` \((\d+) bytes\)$")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    args = ap.parse_args()
    root = Path(args.root)
    manifest = Path(__file__).resolve().parents[2] / "docs" / "release_manifest.md"
    bad = 0
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        m = LINE.match(line.strip())
        if not m:
            continue
        rel, want_sha, want_size = m.group(1), m.group(2), int(m.group(3))
        fp = root / rel
        checked += 1
        if not fp.is_file():
            print(f"MISSING {rel}")
            bad += 1
        elif fp.stat().st_size != want_size or sha(fp) != want_sha:
            print(f"MISMATCH {rel} (tree: {fp.stat().st_size} bytes {sha(fp)[:12]}...)")
            bad += 1
    print(f"verify_manifest: {checked - bad}/{checked} OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
