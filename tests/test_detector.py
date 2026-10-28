import cv2
import numpy as np
import pytest

from syateki import config
from syateki.detector import blur_gray, detect_hit, score_for_distance, score_for_point

W, H = config.PROJ_W, config.PROJ_H
CX, CY = config.TARGET_CENTER
BLUE = (200, 80, 20)     # HSV H≒108
ORANGE = (0, 160, 255)   # HSV H≒19, 背景より明るい


@pytest.mark.parametrize("d,score", [
    (0, 100), (60, 100), (60.01, 50), (150, 50), (150.01, 10), (260, 10), (260.01, 0), (500, 0),
])
def test_score_boundaries(d, score):
    assert score_for_distance(d) == score


def test_score_for_point():
    assert score_for_point(CX, CY) == 100
    assert score_for_point(CX + 150, CY) == 50
    assert score_for_point(CX, CY + 261) == 0


def background():
    return np.full((H, W, 3), 120, np.uint8)


def detect(frame):
    return detect_hit(blur_gray(background()), frame)


def test_blue_only_uses_bottom_point():
    f = background()
    cv2.ellipse(f, (CX + 100, CY), (30, 8), 0, 0, 360, BLUE, -1)
    hit, mask = detect(f)
    assert hit is not None
    x, y = hit
    assert abs(x - (CX + 100)) <= 4
    assert CY + 7 <= y <= CY + 14  # 楕円の下端 (+dilate 分)
    assert score_for_point(x, y) == 50
    assert mask.shape == (H, W)


def test_dart_uses_orange_tip():
    # 縦長の青ボディ + 下端にオレンジの先端
    f = background()
    body_c, tip = (CX + 30, CY - 120), (CX + 30, CY - 87)
    cv2.ellipse(f, body_c, (8, 30), 0, 0, 360, BLUE, -1)
    cv2.circle(f, tip, 7, ORANGE, -1)
    hit, _ = detect(f)
    assert hit is not None
    assert abs(hit[0] - tip[0]) <= 3 and abs(hit[1] - tip[1]) <= 3
    assert score_for_point(*hit) == 50


def test_colorless_change_ignored():
    f = background()
    cv2.ellipse(f, (CX, CY), (30, 10), 0, 0, 360, (250, 250, 250), -1)
    cv2.ellipse(f, (CX + 80, CY), (30, 10), 0, 0, 360, (10, 10, 10), -1)
    hit, _ = detect(f)
    assert hit is None


def test_outside_target_ignored():
    f = background()
    cv2.ellipse(f, (CX + 400, CY), (30, 8), 0, 0, 360, BLUE, -1)
    assert detect(f)[0] is None


def test_too_small_ignored():
    f = background()
    f[CY, CX] = BLUE  # 1px
    assert detect(f)[0] is None


def test_no_change_no_hit():
    assert detect(background())[0] is None


def test_offset_applied():
    f = background()
    cv2.ellipse(f, (CX, CY), (30, 8), 0, 0, 360, BLUE, -1)
    base, _ = detect_hit(blur_gray(background()), f, offset=(0, 0))
    moved, _ = detect_hit(blur_gray(background()), f, offset=(5, -10))
    assert moved == (base[0] + 5, base[1] - 10)
