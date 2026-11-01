"""無人運用のラウンド制 (ATTRACT → PLAYING → RESULT → ATTRACT)。"""
import json

import cv2
import numpy as np
import pytest

from syateki import config
from syateki.app import App
from syateki.rounds import ATTRACT, PLAYING, RESULT, RoundState, prize_for, prize_hint

W, H = config.PROJ_W, config.PROJ_H
CX, CY = config.TARGET_CENTER
FULL = [[0, 0], [W, 0], [W, H], [0, H]]
POS = {100: (CX, CY), 50: (CX + 100, CY), 10: (CX + 200, CY)}  # 点数ごとの着弾位置


# 会場で使う既定値 (fixture で差し替える前に控えておく)
DEFAULTS = dict(SHOTS_PER_ROUND=config.SHOTS_PER_ROUND, ROUND_TIMEOUT=config.ROUND_TIMEOUT,
                IDLE_TIMEOUT=config.IDLE_TIMEOUT, RESULT_SEC=config.RESULT_SEC,
                PRIZE_TIERS=list(config.PRIZE_TIERS), AUTO_ROUND=config.AUTO_ROUND)


@pytest.fixture(autouse=True)
def auto_round(monkeypatch):
    """ラウンド制は既定で無効なので、このファイルでは有効にして試す。

    発数・秒数・景品は会場で変わるので、テストは固定値に差し替えて試す (既定値は test_defaults)。
    """
    monkeypatch.setattr(config, "AUTO_ROUND", True)
    for k, v in dict(SHOTS_PER_ROUND=5, ROUND_TIMEOUT=40.0, IDLE_TIMEOUT=15.0, RESULT_SEC=10.0).items():
        monkeypatch.setattr(config, k, v)
    # 景品の点数は会場で変わるので、テストは 300 点で固定して試す
    monkeypatch.setattr(config, "PRIZE_TIERS", [(300, "大きいお菓子", "big"), (0, "お菓子", "normal")])


@pytest.fixture
def played(monkeypatch):
    from syateki import sound
    out = []
    monkeypatch.setattr(sound, "play", out.append)
    return out


def make(tmp_path, mode="normal"):
    p = tmp_path / "corners.json"
    p.write_text(json.dumps(FULL))
    app = App(demo=True, corners_path=str(p), settings_path=str(tmp_path / "settings.json"))
    app.mode = mode
    return app


def bg():
    return np.full((H, W, 3), 120, np.uint8)


def shoot(app, score, now):
    """デモのクリックで着弾させ、クールダウンを終わらせる。"""
    app.on_proj_mouse(cv2.EVENT_LBUTTONDOWN, *POS[score], 0, now=now)
    app.game.update(now + config.COOLDOWN_SEC)


def test_defaults():
    d = DEFAULTS
    assert d["AUTO_ROUND"] is False
    assert (d["SHOTS_PER_ROUND"], d["ROUND_TIMEOUT"], d["IDLE_TIMEOUT"], d["RESULT_SEC"]) == \
        (6, None, None, 15.0)
    assert d["PRIZE_TIERS"][0][:2] == (350, "大きいお菓子")
    assert prize_hint(d["PRIZE_TIERS"], d["SHOTS_PER_ROUND"]) == "6発で350点以上 → 大きいお菓子"


def test_no_idle_timeout_when_none(tmp_path, monkeypatch):
    """IDLE_TIMEOUT = None なら、着弾が途切れても開始から ROUND_TIMEOUT までは続く。"""
    monkeypatch.setattr(config, "IDLE_TIMEOUT", None)
    app = make(tmp_path)
    shoot(app, 50, 1.0)
    assert app.round.time_left(2.0) == pytest.approx(1.0 + config.ROUND_TIMEOUT - 2.0)
    app.step(bg(), 1.0 + config.ROUND_TIMEOUT - 0.01)
    assert app.round.state == PLAYING
    app.step(bg(), 1.0 + config.ROUND_TIMEOUT)
    assert app.round.state == RESULT and app.round.end_reason == "round_timeout"


def test_no_time_limit_when_none(tmp_path, monkeypatch):
    """ROUND_TIMEOUT も IDLE_TIMEOUT も None なら、時間では終わらない (発数だけ)。"""
    monkeypatch.setattr(config, "ROUND_TIMEOUT", None)
    monkeypatch.setattr(config, "IDLE_TIMEOUT", None)
    app = make(tmp_path)
    shoot(app, 50, 1.0)
    assert app.round.time_left(2.0) is None
    app.step(bg(), 10_000.0)
    assert app.round.state == PLAYING


@pytest.mark.parametrize("score,kind", [(0, "normal"), (299, "normal"), (300, "big"), (999, "big")])
def test_prize_boundary(score, kind):
    assert prize_for(score)[1] == kind
    assert prize_for(300)[0] == "大きいお菓子" and prize_for(299)[0] == "お菓子"


def test_prize_hint(monkeypatch):
    assert prize_hint() == "5発で300点以上 → 大きいお菓子"
    assert RoundState().info(0.0)["hint"] == prize_hint()
    monkeypatch.setattr(config, "SHOTS_PER_ROUND", 3)
    monkeypatch.setattr(config, "PRIZE_TIERS", [(500, "特大", "big"), (200, "中", "normal"),
                                                (0, "小", "normal")])
    assert RoundState().info(0.0)["hint"] == "3発で500点以上 → 特大"
    assert prize_hint([(0, "お菓子", "normal")]) is None


def test_prize_none_when_no_tier_matches():
    assert prize_for(50, [(100, "x", "big")]) == (None, None)


def test_five_shots_end_round_then_auto_reset(tmp_path, played):
    app = make(tmp_path)
    assert app.round.state == ATTRACT and app.round.info(0.0)["time_left"] is None
    for i, s in enumerate([100, 50, 50, 10, 100]):
        assert app.round.state == (ATTRACT if i == 0 else PLAYING)
        shoot(app, s, 1.0 + i * 2)
    r = app.round
    assert r.state == RESULT and r.end_reason == "shots" and r.score == 310
    assert r.prize == ("大きいお菓子", "big") and played[-1] == "hit100_fever"
    info = r.info(9.0)
    assert info["state"] == "result" and info["shots"] == 5 and info["max_shots"] == 5
    assert info["prize_kind"] == "big" and info["result_started"] == 9.0
    assert info["time_left"] == pytest.approx(config.RESULT_SEC)

    app.step(bg(), 9.0 + config.RESULT_SEC - 0.1)      # まだ結果表示
    assert r.state == RESULT and app.game.total_score == 310
    app.step(bg(), 9.0 + config.RESULT_SEC)            # 自動リセット
    assert r.state == ATTRACT and app.game.total_score == 0 and app.game.hits == []
    assert r.shots == 0 and r.prize == (None, None)
    app.step(bg(), 20.0)                               # 背景も撮り直されている
    assert app.bg_gray is not None


def test_hits_ignored_during_result(tmp_path, played):
    app = make(tmp_path)
    for i in range(5):
        shoot(app, 10, 1.0 + i)
    assert app.round.state == RESULT and app.game.total_score == 50
    n = len(played)
    shoot(app, 100, 7.0)
    assert app.game.total_score == 50 and len(app.game.hits) == 5 and len(played) == n
    assert app.round.prize[1] == "normal"


def test_round_timeout(tmp_path):
    app = make(tmp_path)
    t = 0.0
    shoot(app, 10, t)                                  # 開始
    for i in range(3):                                 # 10秒おきなら IDLE では切れない
        t += 10.0
        shoot(app, 10, t)
    app.step(bg(), config.ROUND_TIMEOUT - 0.01)
    assert app.round.state == PLAYING
    app.step(bg(), config.ROUND_TIMEOUT)
    assert app.round.state == RESULT and app.round.end_reason == "round_timeout"
    assert app.round.score == 40 and app.round.shots == 4


def test_idle_timeout(tmp_path):
    app = make(tmp_path)
    shoot(app, 50, 5.0)
    shoot(app, 50, 8.0)
    assert app.round.time_left(10.0) == pytest.approx(8.0 + config.IDLE_TIMEOUT - 10.0)
    app.step(bg(), 8.0 + config.IDLE_TIMEOUT - 0.01)
    assert app.round.state == PLAYING
    app.step(bg(), 8.0 + config.IDLE_TIMEOUT)
    assert app.round.state == RESULT and app.round.end_reason == "idle_timeout"
    assert app.round.prize == ("お菓子", "normal")


def test_attract_never_times_out(tmp_path):
    app = make(tmp_path)
    app.step(bg(), 1000.0)
    assert app.round.state == ATTRACT


def test_prize_uses_fever_double_score(tmp_path, played):
    """fever: ブル2連続でボーナス → 以降2倍。素点が 300 未満でも、2倍込みで 300 以上なら大きいお菓子。"""
    app = make(tmp_path, mode="fever")
    for i, s in enumerate([100, 100, 10, 10, 10]):      # 100 + 100 + 10x2 x3 = 260 (素点 230)
        shoot(app, s, 1.0 + i)
    raw = sum(h.score for h in app.game.hits)
    assert raw == 230 and app.game.total_score == 260 and app.round.prize[1] == "normal"

    app.reset_game()
    for i, s in enumerate([100, 100, 50, 10, 10]):      # 200 + 50x2 + 10x2 + 10x2 = 340 (素点 270)
        shoot(app, s, 100.0 + i)
    assert sum(h.score for h in app.game.hits) == 270
    assert app.round.score == 340 and app.round.prize[1] == "big"
    assert app.game.bonus_until is None                 # ラウンド終了でボーナスも終わる


def test_reset_key_returns_to_attract(tmp_path):
    app = make(tmp_path)
    shoot(app, 50, 1.0)
    assert app.round.state == PLAYING
    app.handle_key(ord("r"))
    assert app.round.state == ATTRACT and app.game.total_score == 0
    for i in range(5):
        shoot(app, 10, 2.0 + i)
    app.handle_key(ord("r"))                            # RESULT 中でも R で即待機へ
    assert app.round.state == ATTRACT


def test_projector_and_monitor_get_round_info(tmp_path, monkeypatch):
    from syateki import app as app_mod
    calls = []

    def fake(*a, **kw):
        calls.append(kw.get("round_info"))
        return np.zeros((H, W, 3), np.uint8)
    monkeypatch.setattr(app_mod, "render_projector", fake)
    app = make(tmp_path)
    shoot(app, 50, 1.0)
    _, mon = app.step(bg(), 2.0)
    info = calls[-1]
    assert info["state"] == "playing" and info["shots"] == 1
    assert info["time_left"] == pytest.approx(1.0 + config.IDLE_TIMEOUT - 2.0)
    assert app.round.status(2.0).startswith("ROUND: 1/5")


def test_projector_without_round_info_falls_back(tmp_path, monkeypatch):
    from syateki import app as app_mod

    def old(w, h, center, radii, total, hits, cd, calibrating, now, mode="normal",
            bonus_start=None, bonus_until=None):
        return np.zeros((h, w, 3), np.uint8)
    monkeypatch.setattr(app_mod, "render_projector", old)
    app = make(tmp_path)
    proj, _ = app.step(bg(), 0.0)
    assert proj.shape == (H, W, 3)


def test_auto_round_false_keeps_old_behavior(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_ROUND", False)
    app = make(tmp_path)
    assert app.round is None
    for i in range(8):                                  # 5発を超えても貯まり続ける
        shoot(app, 10, 1.0 + i * 2)
    app.step(bg(), 500.0)                               # 時間がたってもリセットされない
    assert app.game.total_score == 80 and len(app.game.hits) == 8


# ---------------------------------------------------------------- U キー
def test_u_key_toggles_saves_and_restores(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "AUTO_ROUND", False)      # 初期値 OFF
    app = make(tmp_path)
    assert app.round is None and not app.auto_round
    shoot(app, 50, 1.0)
    assert app.handle_key(ord("u"))                      # ON: リセットして待機
    assert app.auto_round and app.round.state == ATTRACT and app.game.total_score == 0
    assert json.loads((tmp_path / "settings.json").read_text())["auto_round"] is True
    assert make(tmp_path).auto_round                      # 再起動しても ON (config より優先)

    for i in range(config.SHOTS_PER_ROUND):               # ON ならラウンドが動く
        shoot(app, 10, 2.0 + i)
    assert app.round.state == RESULT
    assert app.handle_key(ord("U"))                      # OFF: リセットして通常へ
    assert app.round is None and app.game.total_score == 0
    assert json.loads((tmp_path / "settings.json").read_text())["auto_round"] is False
    assert not make(tmp_path).auto_round
    for i in range(config.SHOTS_PER_ROUND + 2):           # OFF なら貯まり続ける
        shoot(app, 10, 20.0 + i * 2)
    app.step(bg(), 500.0)
    assert app.game.total_score == 10 * (config.SHOTS_PER_ROUND + 2)


def test_round_info_only_when_auto_on(tmp_path, monkeypatch):
    from syateki import app as app_mod
    calls = []
    monkeypatch.setattr(app_mod, "render_projector",
                        lambda *a, **kw: calls.append(kw) or np.zeros((H, W, 3), np.uint8))
    app = make(tmp_path)
    app.step(bg(), 0.0)
    assert calls[-1]["round_info"]["state"] == "attract"
    app.handle_key(ord("u"))
    app.step(bg(), 1.0)
    assert "round_info" not in calls[-1]
