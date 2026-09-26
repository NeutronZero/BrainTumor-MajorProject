"""fetch_models.py — verify pinned checkpoints, downloading from Hub if needed.

Sources, in order: local files (incl. `git lfs pull` materialization),
then Hugging Face Hub. Every downloaded file is SHA256-verified against
the frozen release manifest before use.

Run: python scripts/fetch_models.py [--check-only]
Requires for download: `pip install huggingface-hub` (fetch-only dep,
kept out of requirements.lock so the runtime image stays lean).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HF_REPO = "NeutronZero/BrainTumor-MajorProject-REL002"

# local path -> (Hub filename, SHA256 from docs/release_manifest.md, bytes)
EXPECTED = {
    "checkpoints/CLS-001/best.pt": (
        "CLS-001-best.pt",
        "451e4fc4b12446764262c159253e7550b5a08db49bb2956059f71613e268095e",
        111358647,
    ),
    "checkpoints/SEG-001/best.pt": (
        "SEG-001-best.pt",
        "ce29df5e4225ed8e7aa0fb29599e53c6abd7a9e27d5bbb2d7f35a3332a2d1b80",
        124231879,
    ),
    "checkpoints/ROB-001/best.pt": (
        "ROB-001-best.pt",
        "36091d63cea9af94c568ee3eaf9895658c7eae9a7179cf1fee6c397f1ad82620",
        372593293,
    ),
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _valid(fp: Path, sha: str, size: int) -> bool:
    return (
        fp.is_file()
        and fp.stat().st_size == size
        and sha256_file(fp) == sha
    )


def _download(repo: str, filename: str, dest: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit(
            "fetch_models: need `pip install huggingface-hub` to download weights "
            f"(or run `git lfs pull` / copy {dest} manually)."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    got = hf_hub_download(repo_id=repo, filename=filename, local_dir=dest.parent)
    Path(got).replace(dest)


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify or fetch pinned checkpoints.")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--repo", default=HF_REPO)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    rc = 0
    for local, (hub_name, sha, size) in EXPECTED.items():
        fp = root / local
        if _valid(fp, sha, size):
            print(f"fetch_models: OK {local}")
            continue
        if args.check_only:
            print(f"fetch_models: MISSING/BAD {local} — run without --check-only to fetch.")
            rc = 1
            continue
        print(f"fetch_models: fetching {hub_name} ({size // 1024 // 1024}MB)...")
        _download(args.repo, hub_name, fp)
        if not _valid(fp, sha, size):
            print(f"fetch_models: HASH MISMATCH after download: {local}")
            return 1
        print(f"fetch_models: OK {local} (verified)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
