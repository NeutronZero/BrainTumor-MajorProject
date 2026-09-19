"""audit_near_duplicates.py — lightweight aHash proxy (report-only)."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gate.common import write_json  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _ahash(p: Path):
    from PIL import Image
    with Image.open(p) as im:
        g = im.convert("L").resize((8, 8))
        px = list(g.getdata())
        avg = sum(px) / len(px)
        return "".join("1" if v > avg else "0" for v in px)


def main() -> dict:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    files = [p for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXTS] if data_dir.exists() else []
    try:
        import PIL  # noqa: F401
        buckets: dict[str, list[str]] = defaultdict(list)
        for p in sorted(files):
            try:
                buckets[_ahash(p)].append(str(p.relative_to(root)))
            except Exception:
                pass
        groups = {h: v for h, v in buckets.items() if len(v) > 1}
        result = {"check": "near_duplicates", "pass": True, "policy": "report_only",
                  "n_files": len(files), "n_collision_groups": len(groups),
                  "groups_sample": list(groups.values())[:10]}
    except ImportError:
        result = {"check": "near_duplicates", "pass": True, "policy": "report_only",
                  "skipped": "PIL unavailable", "n_files": len(files)}
    write_json("near_duplicate_report.json", result)
    print(f"near-duplicates: report-only n={len(files)}")
    return result


if __name__ == "__main__":
    main()
