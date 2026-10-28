import json

import cv2
import numpy as np

from syateki import config
from syateki.app import App
from syateki.game import COOLDOWN, READY
from syateki.keys import decode_key

W, H = config.PROJ_W, config.PROJ_H
FULL = [[0, 0], [W, 0], [W, H], [0, H]]  # 恒等変換になる四隅


def make_app(tmp_path, demo=False):
    p = tmp_path / "corners.json"
    p.write_text(json.dumps(FULL))
    return App(demo=demo, corners_path=str(p)), p


def test_decode_keys():
    assert decode_key(-1) is None
    assert decode_key(0x260000) == "up"       # Windows
    assert decode_key(0x250000) == "left"
    assert decode_key(0xFF52) == "up"         # GTK
    assert decode_key(0xFF51 | (0x10 << 16)) == "left"  # GTK + NumLock
    assert decode_key(0x1000014) == "right"   # Qt
    assert decode_key(ord("Q")) == "q"
    assert decode_key(ord("a")) == "a"
    assert decode_key(27) == "esc"
    assert decode_key(9) == "tab"
    # 旧版の & 0xFF で化けていたコードが文字にならない
    assert decode_key(0xFF52) != "r" and decode_key(0xFF51) != "q"


def test_pipeline_detects_dart_and_cooldown(tmp_path):
    app, _ = make_app(tmp_path)
    bg = np.full((H, W, 3), 120, np.uint8)
    t = 1000.0
    app.step(bg, t)  # 背景を撮る
    dart = bg.copy()
    cv2.ellipse(dart, (config.TARGET_CENTER[0], config.TARGET_CENTER[1] - 10), (8, 25), 0, 0, 360,
                (200, 80, 20), -1)
    proj, mon = app.step(dart, t + 0.03)
    assert app.game.total_score == 100 and app.game.phase == COOLDOWN
    assert proj.shape == (H, W, 3) and mon.shape == (H, W, 3)
    app.step(dart, t + 0.5)  # COOLDOWN 中は検出しない
    assert app.game.total_score == 100
    app.step(dart, t + 0.03 + config.COOLDOWN_SEC)  # 終了: 今の映像を背景にする
    assert app.game.phase == READY
    app.step(dart, t + 2.0)  # 弾が刺さったままでも二重計上しない
    assert app.game.total_score == 100 and len(app.game.hits) == 1


def test_arrow_and_wasd_nudge(tmp_path):
    app, p = make_app(tmp_path)
    app.selected = 1
    for code, d in [(0x260000, (0, -2)), (0xFF54, (0, 2)), (ord("a"), (-2, 0)), (0x270000, (2, 0))]:
        before = list(app.corners[1])
        assert app.handle_key(code)
        assert app.corners[1] == [before[0] + d[0], before[1] + d[1]]
    assert json.loads(p.read_text())[1] == app.corners[1]
    assert app.handle_key(9) and app.selected == 2
    assert not app.handle_key(27)
    assert not app.handle_key(ord("Q"))


def test_drag_saves_only_on_release(tmp_path):
    app, p = make_app(tmp_path)
    app.on_camera_mouse(cv2.EVENT_LBUTTONDOWN, 20, 600, 0)
    app.on_camera_mouse(cv2.EVENT_MOUSEMOVE, 30, 610, cv2.EVENT_FLAG_LBUTTON)
    assert json.loads(p.read_text()) == FULL  # まだ保存しない
    assert app.corners[3] == [30, 610]
    app.on_camera_mouse(cv2.EVENT_LBUTTONUP, 30, 610, 0)
    assert json.loads(p.read_text())[3] == [30, 610]


def test_buttons_and_click4(tmp_path):
    from syateki.render_monitor import BUTTONS
    app, p = make_app(tmp_path)
    x1, y1, x2, y2 = BUTTONS["click4"]
    app.on_camera_mouse(cv2.EVENT_LBUTTONDOWN, (x1 + x2) // 2, (y1 + y2) // 2, 0)
    assert app.click_step == 0
    pts = [[50, 60], [600, 55], [610, 450], [40, 440]]
    for x, y in pts:
        app.on_camera_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0)
    assert app.click_step == -1 and json.loads(p.read_text()) == pts
    x1, y1, x2, y2 = BUTTONS["auto"]
    app.on_camera_mouse(cv2.EVENT_LBUTTONDOWN, x1 + 2, y1 + 2, 0)
    assert app.calib_step == 1


def test_demo_click_scores(tmp_path):
    app, _ = make_app(tmp_path, demo=True)
    cx, cy = config.TARGET_CENTER
    app.on_proj_mouse(cv2.EVENT_LBUTTONDOWN, cx + 100, cy, 0, now=10.0)
    assert app.game.total_score == 50
    app.on_proj_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0, now=10.5)  # COOLDOWN 中は無視
    assert app.game.total_score == 50
    app.game.update(12.0)
    app.on_proj_mouse(cv2.EVENT_LBUTTONDOWN, 5, 5, 0, now=12.1)  # 的の外
    assert app.game.total_score == 50 and app.game.ready
    app.on_proj_mouse(cv2.EVENT_RBUTTONDOWN, 0, 0, 0)
    assert app.game.total_score == 0


def test_auto_calibration_flow(tmp_path):
    app, p = make_app(tmp_path)
    cam = np.full((480, 640, 3), 30, np.uint8)
    app.start_auto_calibration(now=0.0)
    proj, _ = app.step(cam, 0.0)        # 暗フレーム
    proj, _ = app.step(cam, 0.1)        # 白点灯中
    assert proj.mean() > 250
    lit = cam.copy()
    quad = [(100, 70), (540, 80), (560, 410), (90, 400)]
    cv2.fillPoly(lit, [np.int32(quad)], (230, 230, 230))
    app.step(lit, 0.3)
    assert app.calib_step == 0
    for (fx, fy), (qx, qy) in zip(app.corners, quad):
        assert abs(fx - qx) <= 6 and abs(fy - qy) <= 6
    assert json.loads(p.read_text()) == app.corners
