import json

import numpy as np
import cv2
import pytest

from syateki.calibration import detect_projection_corners, load_corners, order_points, save_corners


def test_order_points_shuffled():
    pts = [[500, 400], [100, 80], [90, 420], [520, 60]]
    assert order_points(pts) == [[100, 80], [520, 60], [500, 400], [90, 420]]


def test_order_points_rectangle():
    pts = [[0, 0], [10, 10], [10, 0], [0, 10]]
    assert order_points(pts) == [[0, 0], [10, 0], [10, 10], [0, 10]]


@pytest.mark.parametrize("quad", [
    [(120, 80), (520, 100), (560, 420), (80, 400)],   # 台形
    [(60, 50), (590, 50), (590, 430), (60, 430)],     # 長方形
])
def test_flash_calibration_finds_corners(quad):
    dark = np.full((480, 640), 30, np.uint8)
    cv2.circle(dark, (300, 250), 40, 60, -1)  # 暗フレーム側の模様は無視される
    bright = dark.copy()
    cv2.fillPoly(bright, [np.int32(quad)], 230)
    found = detect_projection_corners(dark, bright)
    assert found is not None
    for (fx, fy), (qx, qy) in zip(found, quad):
        assert abs(fx - qx) <= 6 and abs(fy - qy) <= 6


def test_flash_calibration_no_change():
    dark = np.full((480, 640), 30, np.uint8)
    assert detect_projection_corners(dark, dark.copy()) is None


def test_flash_calibration_too_small():
    dark = np.full((480, 640), 30, np.uint8)
    bright = dark.copy()
    cv2.rectangle(bright, (100, 100), (160, 160), 230, -1)  # 面積 < 8000
    assert detect_projection_corners(dark, bright) is None


def test_corners_roundtrip(tmp_path):
    p = tmp_path / "corners.json"
    save_corners([[1, 2], [3, 4], [5, 6], [7, 8]], str(p))
    assert json.loads(p.read_text()) == [[1, 2], [3, 4], [5, 6], [7, 8]]
    assert load_corners(str(p)) == [[1, 2], [3, 4], [5, 6], [7, 8]]


def test_corners_missing_or_broken(tmp_path):
    d = [[0, 0], [1, 0], [1, 1], [0, 1]]
    assert load_corners(str(tmp_path / "none.json"), d) == d
    bad = tmp_path / "bad.json"
    bad.write_text("{oops")
    assert load_corners(str(bad), d) == d
