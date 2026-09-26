import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.data.brisc import CLASS_TO_IDX, ClsDataset, SegDataset, load_manifest  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "outputs" / "data_gate_0" / "project_manifest.csv"

import pytest

DATA_PRESENT = (ROOT / "data" / "brisc2025" / "classification_task").is_dir()
needs_data = pytest.mark.skipif(
    not DATA_PRESENT, reason="BRISC bytes not present (external dataset)"
)


def test_manifest_split_counts():
    rows = load_manifest(MANIFEST)
    pool = [r for r in rows if r["population"] == "official_train_pool"]
    assert len([r for r in pool if r["project_split"] == "train"]) == 4000
    assert len([r for r in pool if r["project_split"] == "val"]) == 1000
    assert set(CLASS_TO_IDX) == {"glioma", "meningioma", "pituitary", "notumor"}
    assert not [
        r for r in pool if r["project_split"] == "val" and r["cross_split_contaminated"] == "1"
    ]


@needs_data
def test_cls_dataset_getitem():
    rows = load_manifest(MANIFEST)
    ds = ClsDataset(ROOT, rows, "val")
    assert len(ds) == 1000
    img, label = ds[0]
    assert label in (0, 1, 2, 3)


@needs_data
def test_seg_dataset_split_rule():
    rows = load_manifest(MANIFEST)
    tr, va = SegDataset(ROOT, rows, "train"), SegDataset(ROOT, rows, "val")
    assert len(tr) > 3000 and len(va) > 500
    img, mask = va[0]
    assert img.size == mask.size
    stems_tr = {p.stem for p, _ in tr.pairs}
    stems_va = {p.stem for p, _ in va.pairs}
    assert not (stems_tr & stems_va)
    cnt = Counter(
        r["project_split"]
        for r in rows
        if r["population"] == "official_train_pool" and Path(r["path"]).stem in stems_va
    )
    assert set(cnt) == {"val"}
