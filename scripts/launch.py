"""launch.py — one-command offline launcher (§72). Assumes fetch_models done."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    print("Launching FastAPI (offline inference, no network required)...")
    return subprocess.call(
        [sys.executable, "-m", "uvicorn", "app.api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=root,
    )


if __name__ == "__main__":
    sys.exit(main())
