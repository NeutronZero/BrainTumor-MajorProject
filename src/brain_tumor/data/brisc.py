"""BRISC dataset adapter — reads the FROZEN manifest/split, never the old project."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset

CLASS_TO_IDX = {"glioma": 0, "meningioma": 1, "pituitary": 2, "notumor": 3}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


REQUIRED_COLUMNS = {"path", "sha256", "population", "class",
                    "cross_split_contaminated", "project_split", "sensitivity"}


def load_manifest(manifest_csv: Path) -> list[dict]:
    """Fail-closed manifest load (prior B7 lesson): UTF-8-SIG tolerant, column
    + hard-count validated against the frozen split hashes."""
    with open(manifest_csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    missing = REQUIRED_COLUMNS - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    for r in rows:
        # Manifest paths are stored POSIX-style; normalize Windows separators
        # (v4 kernel evidence: backslashes become one literal segment on Linux).
        r["path"] = r["path"].replace("\\", "/")
        if not r["path"] or not r["class"]:
            raise ValueError("manifest has empty path/class entries")
        if r["class"] not in CLASS_TO_IDX:
            raise ValueError(f"unseen label {r['class']!r} (fail-closed, cf. B7)")
    return rows


def _open_rgb(path: Path):
    from PIL import Image
    return Image.open(path).convert("RGB")


class ClsDataset(Dataset):
    """Classification rows with project_split train/val from frozen manifest."""

    def __init__(self, project_root: Path, manifest_rows: list[dict], split: str,
                 transform=None):
        if split not in ("train", "val"):
            raise ValueError(f"unknown split {split!r} (expected train|val)")
        self.root = project_root
        self.rows = [r for r in manifest_rows
                     if r["population"] == "official_train_pool" and r["project_split"] == split]
        if not self.rows:
            raise ValueError(f"empty {split} set")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = _open_rgb(self.root / r["path"])
        if self.transform:
            img = self.transform(img)
        return img, CLASS_TO_IDX[r["class"]]


class SegDataset(Dataset):
    """Segmentation pairs from seg-train whose stems map into project train/val.

    Mapping key: image stem shared with classification filenames. Split rule is
    frozen in configs/experiment/SEG-001.yaml (seg_val_rule / seg_train_rule).
    """

    def __init__(self, project_root: Path, manifest_rows: list[dict], split: str,
                 pair_transform=None, subset_stems: set[str] | None = None):
        if split not in ("train", "val"):
            raise ValueError(f"unknown split {split!r} (expected train|val)")
        stem_to_split = {Path(r["path"]).stem: r["project_split"] for r in manifest_rows
                         if r["population"] == "official_train_pool"}
        release = project_root / "data" / "brisc2025"
        pairs = []
        skipped = {"no_manifest_entry": 0, "missing_mask": 0, "wrong_split": 0}
        img_dir = release / "segmentation_task" / "train"
        for img_p in sorted((img_dir / "images").glob("*.jpg")):
            if subset_stems is not None and img_p.stem not in subset_stems:
                continue
            m_p = img_dir / "masks" / (img_p.stem + ".png")
            if img_p.stem not in stem_to_split:
                skipped["no_manifest_entry"] += 1
                continue
            if not m_p.exists():
                skipped["missing_mask"] += 1
                continue
            if stem_to_split[img_p.stem] != split:
                skipped["wrong_split"] += 1
                continue
            pairs.append((img_p, m_p))
        if not pairs:
            raise ValueError(f"empty seg {split} set")
        print(f"SegDataset({split}): pairs={len(pairs)} skipped={skipped}")
        self.pairs = pairs
        self.pair_transform = pair_transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        from PIL import Image
        img_p, m_p = self.pairs[i]
        img = Image.open(img_p).convert("L")
        mask = Image.open(m_p).convert("L")
        if self.pair_transform:
            img, mask = self.pair_transform(img, mask)
        return img, mask


def stratified_sampler_weights(labels: list[int], generator=None):
    """Stratified sampling (plan S35 start): inverse-frequency draw weights."""
    from collections import Counter
    counts = Counter(labels)
    w = [1.0 / counts[l] for l in labels]
    return torch.utils.data.WeightedRandomSampler(w, len(w), generator=generator)
