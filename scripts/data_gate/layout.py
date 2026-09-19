"""BRISC dual-tree layout discovery — Gate 0 Revision A.

Release layout (verified from actual bytes, not assumed):
  <release>/classification_task/{train,test}/<class>/*.jpg
  <release>/segmentation_task/{train,test}/{images/*.jpg, masks/*.png}
  <release>/manifest.{csv,json} (+ .sha256), README.md

No component outside this module hardcodes data paths.
"""

from __future__ import annotations

from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLS_IMG_EXTS = {".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Raw folder names observed in release -> frozen contract vocabulary.
CLASS_ALIAS = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
    "no_tumor": "notumor",
    "notumor": "notumor",
}
KNOWN_RAW_LABELS = set(CLASS_ALIAS)


def find_release_root(project_root: Path) -> Path | None:
    cand = project_root / "data" / "brisc2025"
    if (cand / "classification_task").is_dir():
        return cand
    if (project_root / "data" / "classification_task").is_dir():
        return project_root / "data"
    return None


def cls_split_dirs(release: Path) -> dict[str, Path]:
    return {
        "train": release / "classification_task" / "train",
        "test": release / "classification_task" / "test",
    }


def seg_split_dirs(release: Path) -> dict[str, dict[str, Path]]:
    return {
        s: {
            "images": release / "segmentation_task" / s / "images",
            "masks": release / "segmentation_task" / s / "masks",
        }
        for s in ("train", "test")
    }


def list_files(d: Path, exts: set[str]) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.is_file() and p.suffix.lower() in exts)
