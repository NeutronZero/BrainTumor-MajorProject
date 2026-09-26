import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.classification.models import build_classifier, state_for_ckpt  # noqa: E402


def test_build_random_init_offline():
    import torch

    m = build_classifier("convnext_tiny", pretrained=False)
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 4)
    sd = state_for_ckpt(m)
    assert any("classifier" in k for k in sd)


def test_unknown_arch_rejected():
    import pytest

    with pytest.raises(ValueError):
        build_classifier("not_a_model")
