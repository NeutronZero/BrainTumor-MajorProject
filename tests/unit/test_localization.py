import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from brain_tumor.localization.extract import extract  # noqa: E402


def test_empty_mask():
    loc, w = extract(np.zeros((32, 32)), (64, 64))
    assert loc.bbox is None and loc.area_pixels == 0 and w == []


def test_single_component_scaled():
    pm = np.zeros((10, 10))
    pm[2:5, 3:7] = 0.9
    loc, w = extract(pm, (20, 20))  # 2x scale
    assert loc.bbox == (6, 4, 14, 10)
    assert loc.area_pixels == 12
    assert w == []


def test_multi_component_warns_and_largest_wins():
    pm = np.zeros((20, 20))
    pm[1:3, 1:3] = 0.9  # area 4 < min_area 10 -> dropped
    pm[10:15, 10:15] = 0.9  # area 25
    pm[0:5, 15:20] = 0.9  # area 25 (tie -> first wins, still nonempty)
    loc, w = extract(pm, (20, 20))
    assert loc.area_pixels == 50
    assert w == ["multiple_components"]


def test_below_min_area_is_empty():
    pm = np.zeros((10, 10))
    pm[0:2, 0:2] = 0.9
    loc, _ = extract(pm, (10, 10))
    assert loc.area_pixels == 0 and loc.bbox is None
