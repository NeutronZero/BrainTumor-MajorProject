"""launch.py — one-command local runner (API, Streamlit UI, smoke check).

Usage (from repo root):
  python scripts/launch.py          # FastAPI on 127.0.0.1:8000
  python scripts/launch.py api      # same as above
  python scripts/launch.py ui       # Streamlit on localhost:8501
  python scripts/launch.py check    # golden-output smoke test (<30s, no data needed)

Checkpoints are loaded from checkpoints/ when present (git-lfs files);
without them the UI/API start but inference reports unavailable/failed.
Run `python scripts/fetch_models.py` first to verify checkpoint state.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Run API, Streamlit UI, or smoke check.")
    ap.add_argument("target", nargs="?", default="api", choices=["api", "ui", "check"])
    args = ap.parse_args()
    if args.target == "ui":
        print("Launching Streamlit (http://localhost:8501)...")
        return subprocess.call(
            [sys.executable, "-m", "streamlit", "run", "app/streamlit/app.py"],
            cwd=root,
        )
    if args.target == "check":
        print("Running reproducibility smoke check (no data, no network)...")
        return subprocess.call(
            [sys.executable, "scripts/reproducibility_check.py"],
            cwd=root,
        )
    print("Launching FastAPI (offline inference, no network required)...")
    print("Docs: http://127.0.0.1:8000/docs  Health: http://127.0.0.1:8000/health")
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=root,
    )


if __name__ == "__main__":
    sys.exit(main())
