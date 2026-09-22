"""Regression: Grad-CAM target-layer resolution must be multi-arch safe.

Pre-fix behavior raised RuntimeError for any model without a Conv2d under
`features`. Frozen requirement: ConvNeXt-Tiny still resolves
`features.7.2.block.0`; non-`features` architectures fall back to the last
nn.Conv2d anywhere; RuntimeError only when the model has zero Conv2d.
"""

import sys
from pathlib import Path

import pytest
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.explain.gradcam import resolve_target_layer  # noqa: E402


class _FeaturesModel(nn.Module):
    """Mini ConvNeXt-like layout: Conv2d under features.*."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 4, 3),
            nn.Conv2d(4, 4, 3),
        )
        self.head = nn.Linear(4, 2)


class _NoFeaturesModel(nn.Module):
    """Architecture whose Conv2d lives outside `features`."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3)
        self.blocks = nn.Sequential(
            nn.Conv2d(4, 4, 3),
            nn.Conv2d(4, 8, 3),
        )
        self.head = nn.Linear(8, 2)


class _NoConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)


def test_features_layout_takes_priority():
    m = _FeaturesModel()
    name, mod = resolve_target_layer(m)
    assert name == "features.1"  # last Conv2d under features
    assert mod is m.features[1]


def test_non_features_model_falls_back_to_last_conv2d():
    m = _NoFeaturesModel()
    name, mod = resolve_target_layer(m)
    assert name == "blocks.1"  # last Conv2d anywhere (no features.* Conv2d)
    assert mod is m.blocks[1]


def test_model_without_conv2d_raises_runtime_error():
    with pytest.raises(RuntimeError, match="no nn.Conv2d"):
        resolve_target_layer(_NoConvModel())


def test_frozen_convnext_resolves_features_7_2_block_0():
    from brain_tumor.classification.models import build_classifier
    m = build_classifier("convnext_tiny")
    name, mod = resolve_target_layer(m)
    assert name == "features.7.2.block.0"
    assert isinstance(mod, nn.Conv2d)
