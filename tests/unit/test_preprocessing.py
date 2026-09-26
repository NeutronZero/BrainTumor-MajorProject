"""Preprocessing parity (§29): eval/infer path deterministic + shared impl."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.preprocessing.pipeline import (  # noqa: E402
    build_cls_transform,
    build_seg_pair_transform,
)


def _mri():
    import numpy as np
    from PIL import Image

    rng = np.random.RandomState(0)
    return Image.fromarray((rng.rand(300, 260) * 255).astype("uint8"), mode="L")


def test_cls_eval_parity():
    img = _mri().convert("RGB")
    a = build_cls_transform(False)(img)
    b = build_cls_transform(False)(img)
    assert torch.equal(a, b)
    assert a.shape == (3, 224, 224)


def test_seg_pair_sync_and_binary():
    import numpy as np
    from PIL import Image

    img = _mri()
    m = Image.fromarray((np.array(img) > 100).astype("uint8") * 255, mode="L")
    t = build_seg_pair_transform(False)
    i1, m1 = t(img, m)
    i2, m2 = t(img, m)
    assert torch.equal(m1, m2) and torch.equal(i1, i2)
    assert set(torch.unique(m1).tolist()) <= {0.0, 1.0}
    assert m1.shape[-2:] == (256, 256)


def test_seg_noise_default_off_frozen():
    """ROB-001 factor defaults off: train transform without sigma == SEG-001 behavior."""
    import random

    import numpy as np
    from PIL import Image

    torch.manual_seed(0)
    img = _mri()
    m = Image.fromarray((np.array(img) > 100).astype("uint8") * 255, mode="L")
    t = build_seg_pair_transform(True)
    torch.manual_seed(0)
    random.seed(0)
    i1, m1 = t(img, m)
    torch.manual_seed(0)
    random.seed(0)
    i2, m2 = t(img, m)
    assert torch.equal(i1, i2) and torch.equal(m1, m2)


def test_seg_noise_photometric_only():
    """Train-only noise perturbs the image, never the mask or geometry (§30)."""
    import random

    import numpy as np
    from PIL import Image

    img = _mri()
    m = Image.fromarray((np.array(img) > 100).astype("uint8") * 255, mode="L")
    t = build_seg_pair_transform(True, gauss_noise_sigma=0.05)
    torch.manual_seed(7)
    random.seed(1)
    i1, m1 = t(img, m)
    torch.manual_seed(8)
    random.seed(1)
    i2, m2 = t(img, m)
    assert not torch.equal(i1, i2)  # noise active
    assert torch.equal(m1, m2)  # mask untouched
    assert i1.shape[-2:] == (256, 256) and m1.shape == m2.shape
    te = build_seg_pair_transform(False, gauss_noise_sigma=0.05)
    ie1, _ = te(img, m)
    ie2, _ = te(img, m)
    assert torch.equal(ie1, ie2)  # eval path deterministic even with sigma set
