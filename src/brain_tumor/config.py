"""Versioned config loading with SHA256 hashes — §27 reproducibility."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_dict(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config_with_hash(path: Path) -> tuple[dict, str]:
    cfg = load_yaml(path)
    return cfg, sha256_dict(cfg)
