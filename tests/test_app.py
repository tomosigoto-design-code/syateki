import json

import cv2
import numpy as np
import pytest

from syateki import config
from syateki.app import App
from syateki.game import COOLDOWN, READY
from syateki.keys import decode_key

W, H = config.PROJ_W, config.PROJ_H
FULL = [[0, 0], [W, 0], [W, H], [0, H]]  # 恒等変換になる四隅


def make_app(tmp_path, demo=False):
    p = tmp_path / "corners.json"
    p.write_text(json.dumps(FULL))
    return App(demo=demo, corners_path=str(p), settings_path=str(tmp_path / "settings.json")), p


def bg_frame():
    return np.full((H, W, 3), 120, np.uint8)


def dart_frame(x, y):
    f = bg_frame()
    cv2.ellipse(f, (x, y), (8, 25), 0, 0, 360, (200, 80, 20), -1)
    return f


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
    bg = bg_frame()
    t = 1000.0
    app.step(bg, t)  # 背景を撮る
    dart = dart_frame(config.TARGET_CENTER[0], config.TARGET_CENTER[1] - 10)
    for i in range(config.SETTLE_FRAMES - 1):  # 止まるまでは確定しない
        app.step(dart, t + 0.03 * (i + 1))
        assert app.game.total_score == 0 and app.game.ready and app.tracker.tracking
    t += 0.03 * config.SETTLE_FRAMES
    proj, mon = app.step(dart, t)
    assert app.game.total_score == 100 and app.game.phase == COOLDOWN
    assert app.last_settled.reason == "settled" and app.last_settled.frames == config.SETTLE_FRAMES
    assert proj.shape == (H, W, 3) and mon.shape == (H, W, 3)
    app.step(dart, t + 0.5)  # COOLDOWN 中は検出しない
    assert app.game.total_score == 100
    app.step(dart, t + config.COOLDOWN_SEC)  # 終了: 今の映像を背景にする
    assert app.game.phase == READY
    for i in range(5):  # 弾が刺さったままでも二重計上しない
        app.step(dart, t + 2.0 + i * 0.03)
    assert app.game.total_score == 100 and len(app.game.hits) == 1


def test_pipeline_scores_where_dart_stops(tmp_path):
    """上から動いてきて止まった弾は、最初に見えた位置ではなく止まった位置で判定する。"""
    from syateki.detector import blur_gray, find_hit
    app, _ = make_app(tmp_path)
    cx, cy = config.TARGET_CENTER
    t = 0.0
    app.step(bg_frame(), t)
    # 最初のフレームは 10点圏 (cy-230)、止まるのは 100点圏 (cy-20)
    path = [cy - 230, cy - 170, cy - 110, cy - 50] + [cy - 20] * config.SETTLE_FRAMES
    for y in path:
        t += 0.03
        app.step(dart_frame(cx, y), t)
    assert len(app.game.hits) == 1 and app.game.hits[0].score == 100
    expect, _ = find_hit(blur_gray(bg_frame()), dart_frame(cx, cy - 20), offset=(0, 0))
    assert (app.game.hits[0].x, app.game.hits[0].y) == (expect.x, expect.y)
    assert app.last_settled.reason == "settled" and app.last_settled.frames == len(path)


def test_pipeline_flicker_then_stop(tmp_path):
    """飛行中に1フレーム検出が欠けても、最初の位置で確定せず止まった位置で判定する。"""
    app, _ = make_app(tmp_path)
    cx, cy = config.TARGET_CENTER
    frames = [dart_frame(cx, cy - 230), bg_frame(), dart_frame(cx, cy - 120)] + \
        [dart_frame(cx, cy - 20)] * config.SETTLE_FRAMES
    app.step(bg_frame(), 0.0)
    for i, f in enumerate(frames):
        app.step(f, 0.03 * (i + 1))
    assert app.last_settled.reason == "settled"
    assert len(app.game.hits) == 1 and app.game.hits[0].score == 100


@pytest.mark.parametrize("mode,score", [("last", 10), ("drop", 0)])
def test_pipeline_one_frame_blip(tmp_path, monkeypatch, mode, score):
    """1フレームだけ出て消えた弾 (跳ね返り): last は最後の位置で確定、drop は無効。"""
    monkeypatch.setattr(config, "SETTLE_LOST", mode)
    app, _ = make_app(tmp_path)
    cx, cy = config.TARGET_CENTER
    app.step(bg_frame(), 0.0)
    app.step(dart_frame(cx + 200, cy), 0.03)
    assert app.game.total_score == 0
    for i in range(config.SETTLE_GRACE - 1):  # 猶予中はまだ確定しない
        app.step(bg_frame(), 0.06 + 0.03 * i)
        assert app.game.total_score == 0 and app.tracker.tracking
    app.step(bg_frame(), 0.06 + 0.03 * config.SETTLE_GRACE)
    assert app.game.total_score == score and app.last_settled.reason == "lost"
    assert (app.last_settled.hit is None) == (mode == "drop")
    assert app.game.ready == (mode == "drop")


def test_offset_keys_save_and_reload(tmp_path):
    app, _ = make_app(tmp_path)
    assert app.hit_offset == tuple(config.HIT_OFFSET)
    for key in "kkkl":            # 下に6, 右に2
        assert app.handle_key(ord(key))
    assert app.handle_key(ord("I"))  # 大文字も同じ (上に2)
    assert app.handle_key(ord("j")) and app.handle_key(ord("l"))
    assert app.hit_offset == (2, 4)
    assert json.loads((tmp_path / "settings.json").read_text()) == {"hit_offset": [2, 4]}
    assert app.corners == FULL and app.selected == 0  # 角は動かない
    app2, _ = make_app(tmp_path)
    assert app2.hit_offset == (2, 4)


def test_offset_used_by_detection(tmp_path):
    app, _ = make_app(tmp_path)
    app.set_offset(0, 30)
    cx, cy = config.TARGET_CENTER
    app.step(bg_frame(), 0.0)
    for i in range(config.SETTLE_FRAMES):
        app.step(dart_frame(cx, cy - 10), 0.03 * (i + 1))
    h = app.last_settled.hit
    assert (h.x, h.y) == (int(h.raw[0]), int(h.raw[1] + 30))
    assert app.game.hits[0].y == h.y


def test_settings_load_broken_or_missing(tmp_path):
    from syateki.settings import load_offset, save_offset
    p = tmp_path / "settings.json"
    assert load_offset(str(p)) == tuple(config.HIT_OFFSET)
    p.write_text("{broken")
    assert load_offset(str(p)) == tuple(config.HIT_OFFSET)
    p.write_text(json.dumps({"other": 1}))
    save_offset((-4, 6), str(p))
    assert json.loads(p.read_text()) == {"other": 1, "hit_offset": [-4, 6]}
    assert load_offset(str(p)) == (-4, 6)


def test_one_click_offset_calibration(tmp_path):
    from syateki.app import OFS_IDLE, OFS_WAIT_CLICK, OFS_WAIT_SHOT
    app, _ = make_app(tmp_path)
    cx, cy = config.TARGET_CENTER
    assert app.handle_key(ord("o")) and app.offset_step == OFS_WAIT_SHOT
    app.step(bg_frame(), 0.0)
    for i in range(config.SETTLE_FRAMES):
        app.step(dart_frame(cx, cy - 10), 0.03 * (i + 1))
    assert app.offset_step == OFS_WAIT_CLICK
    rx, ry = app.offset_ref
    app.on_proj_mouse(cv2.EVENT_LBUTTONDOWN, int(rx) + 3, int(ry) - 8, 0)
    assert app.offset_step == OFS_IDLE
    assert app.hit_offset == (round(int(rx) + 3 - rx), round(int(ry) - 8 - ry))
    assert json.loads((tmp_path / "settings.json").read_text())["hit_offset"] == list(app.hit_offset)
    # 補正後の判定点 = クリックした位置
    assert (int(rx + app.hit_offset[0]), int(ry + app.hit_offset[1])) == (int(rx) + 3, int(ry) - 8)
    # O 2回で中止
    app.handle_key(ord("o")), app.handle_key(ord("o"))
    assert app.offset_step == OFS_IDLE


def test_one_click_offset_in_demo(tmp_path):
    app, _ = make_app(tmp_path, demo=True)
    cx, cy = config.TARGET_CENTER
    app.handle_key(ord("o"))
    app.on_proj_mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0, now=1.0)   # 着弾 (デモ)
    app.on_proj_mouse(cv2.EVENT_LBUTTONDOWN, cx + 4, cy + 10, 0, now=1.1)  # 実際の位置
    assert app.hit_offset == (4, 10) and len(app.game.hits) == 1


def test_monitor_shows_tracking_and_offset(tmp_path):
    """監視画面の新しい表示が落ちずに描ける (追跡中・確定後・補正モード)。"""
    app, _ = make_app(tmp_path)
    cx, cy = config.TARGET_CENTER
    app.step(bg_frame(), 0.0)
    _, m1 = app.step(dart_frame(cx, cy), 0.03)   # 追跡中
    assert app.tracker.seen == 1
    app.handle_key(ord("o"))
    _, m2 = app.step(dart_frame(cx, cy), 0.06)
    _, m3 = app.step(dart_frame(cx, cy), 0.09)   # 確定
    assert app.last_settled is not None
    assert not np.array_equal(m1, m3)


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
    from syateki.app import CALIB_ARUCO
    assert app.calib_step == CALIB_ARUCO


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


def _calib_app(tmp_path, monkeypatch):
    from syateki import app as app_mod
    app, p = make_app(tmp_path)
    calls = []
    real = app_mod.render_projector
    monkeypatch.setattr(app_mod, "render_projector",
                        lambda *a, **kw: calls.append(1) or real(*a, **kw))
    return app, p, calls


def test_auto_calibration_aruco(tmp_path, monkeypatch):
    """マーカー画面を出し、0.2秒後から検出。4個見えたフレームが溜まったら確定。"""
    from syateki import app as app_mod
    from test_aruco import QUADS, camera_view
    app, p, calls = _calib_app(tmp_path, monkeypatch)
    quad = QUADS[0]
    cam = camera_view(quad)                    # 明るい会場で斜めから撮った体
    app.start_auto_calibration(now=0.0)
    proj, _ = app.step(cam, 0.1)               # 遅延待ち: まだ検出しない
    assert app.calib_step == app_mod.CALIB_ARUCO and not app.markers.counts
    assert not calls and proj.min() == 0 and proj.mean() > 200   # 白地 + 黒いマーカー
    t = config.ARUCO_WAIT_SEC
    for i in range(config.ARUCO_GOOD_FRAMES - 1):
        _, mon = app.step(cam, t + 0.03 * i)
        assert app.calib_step == app_mod.CALIB_ARUCO and len(app.seen_markers) == 4
    app.step(cam, t + 0.1)
    assert app.calib_step == app_mod.CALIB_IDLE
    for (fx, fy), (qx, qy) in zip(app.corners, quad):
        assert abs(fx - qx) <= 3 and abs(fy - qy) <= 3
    assert json.loads(p.read_text()) == app.corners
    assert app.calib_note[0].startswith("AUTO ALIGN OK (ARUCO 4/4)")
    # 直後はキャリブ画面の残像が写るので、少しの間は背景を撮り直し続けて検出しない
    app.step(cam, t + 0.15)
    assert app.tracker.seen == 0 and app.game.total_score == 0


def test_auto_calibration_aruco_two_hidden_uses_partial_on_timeout(tmp_path, monkeypatch):
    from syateki import app as app_mod
    from test_aruco import QUADS, camera_view
    app, _, _ = _calib_app(tmp_path, monkeypatch)
    quad = QUADS[0]
    cam = camera_view(quad, hide=(1, 3))
    app.start_auto_calibration(now=0.0)
    t = config.ARUCO_WAIT_SEC
    while app.calib_step == app_mod.CALIB_ARUCO:
        app.step(cam, t)
        t += 0.1
    assert t <= config.ARUCO_WAIT_SEC + config.ARUCO_TIMEOUT_SEC + 0.2
    assert app.calib_note[0] == "AUTO ALIGN OK (ARUCO 2/4)"
    for (fx, fy), (qx, qy) in zip(app.corners, quad):
        assert abs(fx - qx) <= 4 and abs(fy - qy) <= 4


def test_auto_calibration_falls_back_to_flash(tmp_path, monkeypatch):
    """マーカーが1個も見えなければ、2秒後に 黒 → 白フラッシュ方式へ。"""
    from syateki import app as app_mod
    app, p, calls = _calib_app(tmp_path, monkeypatch)
    cam = np.full((480, 640, 3), 30, np.uint8)
    quad = [(100, 70), (540, 80), (560, 410), (90, 400)]
    lit = cam.copy()
    cv2.fillPoly(lit, [np.int32(quad)], (230, 230, 230))

    app.start_auto_calibration(now=0.0)
    t_fb = config.ARUCO_WAIT_SEC + config.ARUCO_TIMEOUT_SEC
    for t in np.arange(0.0, t_fb, 0.25):
        app.step(lit, float(t))
        assert app.calib_step == app_mod.CALIB_ARUCO
    _, mon = app.step(lit, t_fb)
    assert app.calib_step == app_mod.CALIB_DARK and app.dark_gray is None
    for dt in (0.0, 0.1, 0.19):                 # 投影は真っ黒、まだ暗フレームは撮らない
        proj, _ = app.step(lit, t_fb + dt)
        assert app.calib_step == app_mod.CALIB_DARK and proj.max() == 0 and not calls
    t1 = t_fb + config.DARK_WAIT_SEC
    proj, _ = app.step(cam, t1)                 # 暗フレーム → 白点灯
    assert app.calib_step == app_mod.CALIB_FLASH and proj.mean() > 250
    app.step(lit, t1 + config.FLASH_WAIT_SEC + 0.05)
    assert app.calib_step == app_mod.CALIB_IDLE
    for (fx, fy), (qx, qy) in zip(app.corners, quad):
        assert abs(fx - qx) <= 6 and abs(fy - qy) <= 6
    assert json.loads(p.read_text()) == app.corners
    assert app.calib_note[0] == "AUTO ALIGN OK (FLASH)"


def test_auto_calibration_both_fail(tmp_path, monkeypatch):
    app, p, _ = _calib_app(tmp_path, monkeypatch)
    before = json.loads(p.read_text())
    cam = np.full((480, 640, 3), 30, np.uint8)  # 何も写らない
    app.start_auto_calibration(now=0.0)
    t = 0.0
    while app.calib_step != 0 and t < 5:
        _, mon = app.step(cam, t)
        t += 0.1
    assert app.calib_step == 0 and app.corners == before == json.loads(p.read_text())
    assert app.calib_note[0].startswith("AUTO ALIGN FAILED")
    assert app._calib_status(t) is not None and app._calib_status(t + 10) is None


# ---------------------------------------------------------------- 演出モード
def test_mode_toggle_saves_and_reloads(tmp_path):
    app, _ = make_app(tmp_path)
    assert app.mode == "normal"
    app.set_offset(2, 4)
    assert app.handle_key(ord("p")) and app.mode == "fever"
    data = json.loads((tmp_path / "settings.json").read_text())
    assert data == {"hit_offset": [2, 4], "mode": "fever"}  # オフセットは消えない
    app2, _ = make_app(tmp_path)
    assert app2.mode == "fever" and app2.hit_offset == (2, 4)
    assert app2.handle_key(ord("P")) and app2.mode == "normal"
    assert make_app(tmp_path)[0].mode == "normal"


def test_mode_load_invalid_falls_back(tmp_path):
    from syateki.settings import load_mode
    p = tmp_path / "settings.json"
    assert load_mode(str(p)) == "normal"
    p.write_text(json.dumps({"mode": "party"}))
    assert load_mode(str(p)) == "normal"
    p.write_text("[1, 2]")
    assert load_mode(str(p)) == "normal"


@pytest.mark.parametrize("mode,x_off,expect", [
    ("normal", 0, "hit100"), ("fever", 0, "hit100_fever"),
    ("fever", 100, "hit50"), ("normal", 100, "hit50"),
])
def test_hit_sound_by_mode(tmp_path, monkeypatch, mode, x_off, expect):
    from syateki import sound
    played = []
    monkeypatch.setattr(sound, "play", played.append)
    app, _ = make_app(tmp_path, demo=True)
    app.mode = mode
    cx, cy = config.TARGET_CENTER
    app.on_proj_mouse(cv2.EVENT_LBUTTONDOWN, cx + x_off, cy, 0, now=1.0)
    assert played == [expect]


def test_projector_receives_mode(tmp_path, monkeypatch):
    from syateki import app as app_mod
    calls = []

    def fake(*args, **kw):
        calls.append(kw.get("mode"))
        return np.zeros((H, W, 3), np.uint8)
    monkeypatch.setattr(app_mod, "render_projector", fake)
    app, _ = make_app(tmp_path)
    app.handle_key(ord("p"))
    app.step(bg_frame(), 0.0)
    assert calls == ["fever"]


def test_projector_without_mode_kwarg_falls_back(tmp_path, monkeypatch):
    from syateki import app as app_mod

    def old(w, h, center, radii, total, hits, cd, calibrating, now):  # mode を受け取れない版
        return np.zeros((h, w, 3), np.uint8)
    monkeypatch.setattr(app_mod, "render_projector", old)
    app, _ = make_app(tmp_path)
    proj, _ = app.step(bg_frame(), 0.0)
    assert proj.shape == (H, W, 3)


# ---------------------------------------------------------------- ボーナスタイム
def _shoot(app, x, y, now):
    app.on_proj_mouse(cv2.EVENT_LBUTTONDOWN, x, y, 0, now=now)
    app.game.update(now + config.COOLDOWN_SEC)


def test_bonus_sounds_and_mode_switch(tmp_path, monkeypatch):
    from syateki import sound
    played = []
    monkeypatch.setattr(sound, "play", played.append)
    app, _ = make_app(tmp_path, demo=True)
    app.handle_key(ord("p"))
    cx, cy = config.TARGET_CENTER
    _shoot(app, cx, cy, 1.0)
    _shoot(app, cx, cy, 3.0)
    assert played == ["hit100_fever", "hit100_fever", "bonus_start"]
    _shoot(app, cx + 100, cy, 5.0)                # ボーナス中: 50 x2、音は通常どおり
    assert app.game.total_score == 300 and played[-1] == "hit50"
    app.handle_key(ord("p"))                      # normal に切替 → 即終了
    assert app.game.bonus_until is None and played[-1] == "bonus_end"


def test_bonus_ends_in_step_and_reset(tmp_path, monkeypatch):
    from syateki import sound
    played = []
    monkeypatch.setattr(sound, "play", played.append)
    app, _ = make_app(tmp_path, demo=True)
    app.mode = "fever"
    cx, cy = config.TARGET_CENTER
    _shoot(app, cx, cy, 1.0), _shoot(app, cx, cy, 3.0)
    _, mon = app.step(bg_frame(), 4.0)            # 監視画面にボーナス表示 (落ちない)
    assert app.game.bonus_active(4.0)
    app.step(bg_frame(), 3.0 + config.BONUS_SEC)
    assert app.game.bonus_until is None and played[-1] == "bonus_end"
    _shoot(app, cx, cy, 20.0), _shoot(app, cx, cy, 22.0)
    app.handle_key(ord("r"))                      # リセットで解除
    assert app.game.bonus_until is None and app.game.total_score == 0


def test_projector_receives_bonus(tmp_path, monkeypatch):
    from syateki import app as app_mod
    calls = []

    def fake(*args, **kw):
        calls.append(kw)
        return np.zeros((H, W, 3), np.uint8)
    monkeypatch.setattr(app_mod, "render_projector", fake)
    app, _ = make_app(tmp_path, demo=True)
    app.mode = "fever"
    cx, cy = config.TARGET_CENTER
    _shoot(app, cx, cy, 1.0), _shoot(app, cx, cy, 3.0)
    app.step(bg_frame(), 4.0)
    assert calls[-1] == {"mode": "fever", "bonus_start": 3.0, "bonus_until": 3.0 + config.BONUS_SEC}


def test_projector_with_mode_only_falls_back(tmp_path, monkeypatch):
    from syateki import app as app_mod
    seen = []

    def mid(w, h, center, radii, total, hits, cd, calibrating, now, mode="normal"):
        seen.append(mode)
        return np.zeros((h, w, 3), np.uint8)
    monkeypatch.setattr(app_mod, "render_projector", mid)
    app, _ = make_app(tmp_path)
    app.mode = "fever"
    proj, _ = app.step(bg_frame(), 0.0)
    assert proj.shape == (H, W, 3) and seen == ["fever"]


def test_bgm_calls(tmp_path, monkeypatch):
    """BGM: 発動で start、延長では止めない、終了・リセット・normal 切替・終了処理で stop。"""
    from syateki import sound
    calls = []
    monkeypatch.setattr(sound, "bgm_start", lambda: calls.append("start"), raising=False)
    monkeypatch.setattr(sound, "bgm_stop", lambda: calls.append("stop"), raising=False)
    app, _ = make_app(tmp_path, demo=True)
    app.mode = "fever"
    cx, cy = config.TARGET_CENTER
    _shoot(app, cx, cy, 1.0), _shoot(app, cx, cy, 2.5)       # 発動 (〜7.5)
    assert calls == ["start"]
    _shoot(app, cx, cy, 4.0), _shoot(app, cx, cy, 5.5)       # 延長 (〜10.5)
    assert calls == ["start"] and app.game.bonus_until == 5.5 + config.BONUS_SEC
    app.step(bg_frame(), 8.0)
    assert calls == ["start"]                                # 延長前の終了時刻では止まらない
    app.step(bg_frame(), 5.5 + config.BONUS_SEC)             # 時間切れ
    assert calls == ["start", "stop"]

    _shoot(app, cx, cy, 20.0), _shoot(app, cx, cy, 21.5)
    app.handle_key(ord("r"))                                 # リセット
    assert calls[-2:] == ["start", "stop"]
    app.handle_key(ord("r"))                                 # ボーナスなしのリセットでは呼ばない
    assert len(calls) == 4

    _shoot(app, cx, cy, 30.0), _shoot(app, cx, cy, 31.5)
    app.handle_key(ord("p"))                                 # normal へ切替
    assert app.mode == "normal" and calls[-2:] == ["start", "stop"] and len(calls) == 6


def test_bgm_missing_in_sound_is_ignored(tmp_path, monkeypatch):
    from syateki import sound
    monkeypatch.delattr(sound, "bgm_start", raising=False)
    monkeypatch.delattr(sound, "bgm_stop", raising=False)
    app, _ = make_app(tmp_path, demo=True)
    app.mode = "fever"
    cx, cy = config.TARGET_CENTER
    _shoot(app, cx, cy, 1.0), _shoot(app, cx, cy, 2.5)
    assert app.game.bonus_active(3.0)
    app.step(bg_frame(), 2.5 + config.BONUS_SEC)
    assert app.game.bonus_until is None
