"""Shared helpers for Data Gate 0 — deterministic, stdlib-first."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def outputs_dir() -> Path:
    d = project_root() / "outputs" / "data_gate_0"
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name: str, obj: dict) -> Path:
    p = outputs_dir() / name
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    return p


def write_csv(name: str, rows: list[dict], fieldnames: list[str]) -> Path:
    p = outputs_dir() / name
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return p
