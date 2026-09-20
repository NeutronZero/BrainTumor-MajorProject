"""EXPL-001 unit tests: Grad-CAM coordinate conventions (pure function, no data)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.explain.gradcam import cam_mass_in_bbox  # noqa: E402


def test_cam_mass_in_bbox_convention():
    """focus_in_bbox coordinate check: all-inside -> 1.0, all-outside -> 0.0,
    None bbox -> None. (Live-check diagnostic for the observed 0.00 value.)"""
    cam = np.zeros((264, 256))
    cam[183:264, 82:170] = 1.0
    assert cam_mass_in_bbox(cam, (82, 183, 170, 264)) == 1.0
    cam2 = np.zeros((264, 256))
    cam2[0:50, 0:50] = 1.0
    assert cam_mass_in_bbox(cam2, (82, 183, 170, 264)) == 0.0
    assert cam_mass_in_bbox(cam, None) is None
