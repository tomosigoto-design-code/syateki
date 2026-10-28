import cv2
import numpy as np
import pytest

from syateki import config
from syateki.detector import (HitTracker, Hit, blur_gray, detect_hit, find_hit,
                              score_for_distance, score_for_point)

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


def _blue_only(blend):
    f = background()
    cv2.ellipse(f, (CX + 100, CY), (30, 8), 0, 0, 360, BLUE, -1)
    hit, mask = find_hit(blur_gray(background()), f, offset=(0, 0), blend=blend)
    assert hit is not None and hit.method == "lowest" and hit.orange_px < config.ORANGE_MIN_PIXELS
    assert abs(hit.x - (CX + 100)) <= 4
    assert mask.shape == (H, W)
    return hit


def test_blue_only_blend_1_uses_bottom_point():
    hit = _blue_only(1.0)
    assert CY + 7 <= hit.y <= CY + 14  # 楕円の下端 (+dilate 分)
    assert score_for_point(hit.x, hit.y) == 50


def test_blue_only_blend_0_uses_centroid():
    assert abs(_blue_only(0.0).y - CY) <= 1


def test_blue_only_default_blend_is_midpoint():
    assert config.FALLBACK_BLEND == 0.5
    lo, hi, mid = _blue_only(0.0), _blue_only(1.0), _blue_only(None)
    assert abs(mid.raw[1] - (lo.raw[1] + hi.raw[1]) / 2) <= 0.5
    f = background()
    cv2.ellipse(f, (CX + 100, CY), (30, 8), 0, 0, 360, BLUE, -1)
    assert detect(f)[0] == (mid.x, mid.y)  # detect_hit も同じ既定値


def test_dart_uses_orange_tip():
    # 縦長の青ボディ + 下端にオレンジの先端
    f = background()
    body_c, tip = (CX + 30, CY - 120), (CX + 30, CY - 87)
    cv2.ellipse(f, body_c, (8, 30), 0, 0, 360, BLUE, -1)
    cv2.circle(f, tip, 7, ORANGE, -1)
    hit, _ = find_hit(blur_gray(background()), f, offset=(0, 0))
    assert hit is not None and hit.method == "orange"
    assert hit.orange_px >= config.ORANGE_MIN_PIXELS
    assert abs(hit.x - tip[0]) <= 3 and abs(hit.y - tip[1]) <= 3
    assert score_for_point(hit.x, hit.y) == 50


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
    hit, _ = find_hit(blur_gray(background()), f, offset=(5, -10))
    assert (hit.x, hit.y) == moved and (int(hit.raw[0]), int(hit.raw[1])) == base


# ---------------------------------------------------------------- 安定待ち
def h(x, y):
    return Hit(x, y, "orange", 30, (float(x), float(y)))


def test_tracker_waits_until_still():
    t = HitTracker(frames=3, px=6, timeout=0.5, lost="last")
    # 上から動いてくる間は確定しない
    for i, y in enumerate([200, 260, 320]):
        assert t.update(h(640, y), i * 0.03) is None
    assert t.tracking and t.seen == 3
    assert t.update(h(641, 352), 0.09) is None   # 止まった (1/3)
    assert t.update(h(642, 354), 0.12) is None   # 2/3
    s = t.update(h(641, 355), 0.15)              # 3/3
    assert s.reason == "settled" and s.frames == 6 and (s.hit.x, s.hit.y) == (641, 355)
    assert not t.tracking


def test_tracker_lost_last_and_drop():
    t = HitTracker(frames=3, lost="last", grace=3)
    assert t.update(None, 0.0) is None and not t.tracking
    assert t.update(h(600, 300), 0.0) is None
    assert t.update(None, 0.03) is None and t.update(None, 0.06) is None  # 猶予中
    s = t.update(None, 0.09)  # 3フレーム見えない → lost
    assert s.reason == "lost" and s.frames == 4 and (s.hit.x, s.hit.y) == (600, 300)

    t = HitTracker(frames=3, lost="drop", grace=3)
    t.update(h(600, 300), 0.0)
    for i in range(2):
        assert t.update(None, 0.03 * (i + 1)) is None
    s = t.update(None, 0.09)
    assert s.reason == "lost" and s.hit is None
    assert t.update(None, 0.12) is None  # 追跡は終わっている


def test_tracker_grace_keeps_tracking_through_flicker():
    """飛行中に1〜2フレーム欠けても lost にせず、止まった位置で settled。"""
    assert config.SETTLE_GRACE == 3
    t = HitTracker(frames=3, px=6, timeout=0.5, lost="last", grace=3)
    seq = [h(640, 150), None, h(640, 260), None, None,   # ちらつき + 大きく飛ぶ
           h(641, 350), None, h(642, 352), h(641, 353)]   # 止まる (欠けても run は保つ)
    out = [t.update(x, i * 0.03) for i, x in enumerate(seq)]
    assert all(o is None for o in out[:-1])
    s = out[-1]
    assert s.reason == "settled" and (s.hit.x, s.hit.y) == (641, 353) and s.frames == len(seq)


def test_tracker_timeout_uses_current_position():
    t = HitTracker(frames=3, px=6, timeout=0.5)
    y, now = 100, 0.0
    s = None
    while s is None:
        s = t.update(h(640, y), now)
        y, now = y + 20, now + 0.1   # 動き続ける
    assert s.reason == "timeout" and s.hit.y == 200 and s.frames == 6
