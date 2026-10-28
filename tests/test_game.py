import pytest

from syateki import config
from syateki.game import GameState


def hit(g, score, now, fever=True):
    return g.register_hit(640, 360, score, now, fever=fever)


def test_two_bulls_in_fever_start_bonus():
    g = GameState()
    assert hit(g, 100, 0.0) == (1, None)
    assert hit(g, 100, 2.0) == (1, "start")   # 発動したブル自体は等倍
    assert g.total_score == 200
    assert g.bonus_start == 2.0 and g.bonus_until == 2.0 + config.BONUS_SEC
    assert g.bull_streak == 0 and g.bonus_active(3.0)


def test_other_score_between_bulls_breaks_streak():
    g = GameState()
    hit(g, 100, 0.0)
    hit(g, 50, 2.0)
    assert hit(g, 100, 4.0) == (1, None)
    assert g.bonus_until is None
    assert hit(g, 100, 6.0) == (1, "start")


def test_normal_mode_never_starts_bonus():
    g = GameState()
    for i in range(4):
        assert hit(g, 100, i * 2.0, fever=False) == (1, None)
    assert g.bonus_until is None and g.total_score == 400


def test_bonus_doubles_and_ends_after_bonus_sec():
    g = GameState()
    hit(g, 100, 0.0), hit(g, 100, 2.0)
    assert hit(g, 50, 4.0) == (2, None)
    assert g.total_score == 200 + 100
    assert g.hits[-1].score == 50                 # 履歴は素点
    if hasattr(g.hits[-1], "mult"):
        assert g.hits[-1].mult == 2
    assert g.bonus_left(4.0) == pytest.approx(config.BONUS_SEC - 2.0)
    assert not g.update_bonus(2.0 + config.BONUS_SEC - 0.01)
    assert config.BONUS_SEC == 5.0
    assert g.update_bonus(2.0 + config.BONUS_SEC)   # 5秒で終了
    assert g.bonus_until is None and not g.update_bonus(20.0)
    assert hit(g, 10, 20.0) == (1, None)


def test_two_more_bulls_extend_bonus():
    g = GameState()
    hit(g, 100, 0.0), hit(g, 100, 2.0)            # 開始 (〜2.0 + BONUS_SEC)
    t1, t2 = 2.0 + config.BONUS_SEC * 0.3, 2.0 + config.BONUS_SEC * 0.8
    assert hit(g, 100, t1) == (2, None)
    assert hit(g, 100, t2) == (2, "extend")
    assert g.bonus_start == 2.0 and g.bonus_until == t2 + config.BONUS_SEC
    assert g.total_score == 200 + 200 + 200


def test_reset_clears_bonus_and_streak():
    g = GameState()
    hit(g, 100, 0.0), hit(g, 100, 2.0)
    hit(g, 100, 4.0)
    g.reset()
    assert g.bonus_until is None and g.bonus_start is None and g.bull_streak == 0
    assert hit(g, 100, 6.0) == (1, None)          # 連続数も0から
