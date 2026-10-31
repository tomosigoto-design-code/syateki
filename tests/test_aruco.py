"""ArUco キャリブ: 合成カメラ画像 (斜め・明るい環境光・ノイズ・ブラー) で四隅を求める。"""
import cv2
import numpy as np
import pytest

from syateki import config
from syateki.aruco_calib import (MarkerAverager, calib_screen, corners_from_markers,
                                 detect_markers, marker_layout)

W, H = config.PROJ_W, config.PROJ_H
QUADS = [
    [(110, 80), (540, 95), (565, 405), (85, 390)],    # 台形 (斜め下から)
    [(150, 60), (600, 120), (560, 440), (60, 380)],   # 回転 + 強い歪み
]


def camera_view(quad, contrast=0.35, ambient=150, noise=4.0, blur=3, seed=0, hide=()):
    """投影したキャリブ画面をカメラで撮った体の画像。hide の id のマーカーは人などで隠れる。"""
    screen = calib_screen()
    for i in hide:
        x, y = marker_layout()[i][0]
        cv2.rectangle(screen, (int(x) - 20, int(y) - 20),
                      (int(x) + config.ARUCO_MARKER_PX + 20, int(y) + config.ARUCO_MARKER_PX + 20),
                      (70, 60, 50), -1)
    m = cv2.getPerspectiveTransform(np.float32([[0, 0], [W, 0], [W, H], [0, H]]), np.float32(quad))
    cam = np.full((480, 640, 3), 90, np.uint8)  # 投影の外の壁
    warped = cv2.warpPerspective(screen, m, (640, 480))
    inside = cv2.warpPerspective(np.full((H, W), 255, np.uint8), m, (640, 480)) > 0
    cam[inside] = warped[inside]
    rng = np.random.default_rng(seed)
    cam = cam.astype(np.float32) * contrast + ambient + rng.normal(0, noise, cam.shape)
    cam = np.clip(cam, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(cam, (blur, blur), 0) if blur else cam


def err(found, quad):
    return float(np.abs(np.array(found, float) - np.array(quad, float)).max())


def test_screen_has_four_markers_with_quiet_zone():
    s = calib_screen()
    assert s.shape == (H, W, 3)
    assert s[:config.ARUCO_MARGIN_PX].min() == 255     # 上端は白 (クワイエットゾーン)
    assert sorted(detect_markers(s)) == [0, 1, 2, 3]


@pytest.mark.parametrize("quad", QUADS)
@pytest.mark.parametrize("contrast,ambient", [(1.0, 0), (0.4, 130), (0.3, 165)])
def test_finds_corners_in_bright_room(quad, contrast, ambient):
    cam = camera_view(quad, contrast, ambient)
    found = detect_markers(cam)
    assert sorted(found) == [0, 1, 2, 3]
    corners = corners_from_markers(found)
    assert corners is not None and err(corners, quad) <= 3


@pytest.mark.parametrize("hide", [(0, 2), (1, 3), (2, 3)])
def test_two_markers_hidden_still_ok(hide):
    quad = QUADS[0]
    found = detect_markers(camera_view(quad, hide=hide))
    assert len(found) == 2 and not set(hide) & set(found)
    assert err(corners_from_markers(found), quad) <= 4


def test_one_marker_is_not_enough():
    found = detect_markers(camera_view(QUADS[0], hide=(0, 1, 2)))
    assert len(found) == 1 and corners_from_markers(found) is None


def test_averaging_over_frames():
    quad = QUADS[1]
    avg = MarkerAverager()
    for seed in range(4):
        avg.add(detect_markers(camera_view(quad, noise=6.0, seed=seed)))
    assert avg.full_frames == 4
    assert err(corners_from_markers(avg.mean()), quad) <= 3
