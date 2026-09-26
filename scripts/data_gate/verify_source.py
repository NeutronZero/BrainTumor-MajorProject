"""verify_source.py — §9 acquisition. FAIL unless source/license/version/SHA recorded."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gate.common import write_json  # noqa: E402


def main() -> dict:
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load(open(root / "configs" / "data" / "brisc2025.yaml"))
    missing = [
        k
        for k in ("source", "license", "version_release", "sha256")
        if str(cfg.get(k, "")).startswith("TO_VERIFY")
    ]
    ok = (not missing) and bool(cfg.get("verified", False))
    result = {
        "check": "source",
        "pass": ok,
        "missing": missing,
        "verified_flag": bool(cfg.get("verified", False)),
    }
    write_json("source_report.json", result)
    print(f"source: {'PASS' if ok else 'FAIL'} missing={missing}")
    return result


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["pass"] else 2)
