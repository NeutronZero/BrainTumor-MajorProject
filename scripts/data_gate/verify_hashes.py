"""verify_hashes.py — existence + readability checks (§9 integrity)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_gate.common import sha256_file, write_json  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> dict:
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    files = (
        [p for p in data_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        if data_dir.exists()
        else []
    )
    unreadable: list[str] = []
    rows = []
    for p in sorted(files):
        try:
            h = sha256_file(p)
            rows.append({"path": str(p.relative_to(root)), "sha256": h, "bytes": p.stat().st_size})
        except Exception as e:  # noqa: BLE001
            unreadable.append(f"{p}: {e}")
    ok = len(files) > 0 and not unreadable
    result = {
        "check": "hashes",
        "pass": ok,
        "n_files": len(files),
        "unreadable": unreadable,
        "reason": None if ok else ("dataset_not_present" if not files else "unreadable_files"),
    }
    write_json("integrity_report.json", result)
    # manifest for reproducibility (§27)
    if rows:
        import csv

        mp = root / "outputs" / "data_gate_0" / "dataset_manifest.csv"
        with open(mp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "sha256", "bytes"])
            w.writeheader()
            w.writerows(rows)
    print(f"hashes: {'PASS' if ok else 'FAIL'} n={len(files)}")
    return result


if __name__ == "__main__":
    r = main()
    sys.exit(0 if r["pass"] else 2)
