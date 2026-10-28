"""ゲームの状態 (スコア・着弾履歴・READY/COOLDOWN・ボーナスタイム)。演出は描画側が hits から計算する。"""
from dataclasses import dataclass, field
from typing import Optional

from . import config
from .render_projector import Hit

READY = "READY"
COOLDOWN = "COOLDOWN"
BONUS_STREAK = 2  # fever でブルがこの回数連続したらボーナスタイム


def _make_hit(x, y, score, now, mult):
    """Hit に mult が無い版の render_projector でも動くように作る。"""
    try:
        return Hit(int(x), int(y), int(score), now, mult=mult)
    except TypeError:
        return Hit(int(x), int(y), int(score), now)


@dataclass
class GameState:
    total_score: int = 0
    hits: list = field(default_factory=list)  # list[Hit], 古い順
    phase: str = READY
    cooldown_end: float = 0.0
    bull_streak: int = 0                  # fever での連続ブル数 (100点以外 / リセットで0)
    bonus_start: Optional[float] = None   # ボーナスタイム (スコア2倍) の開始・終了時刻
    bonus_until: Optional[float] = None

    def reset(self):
        self.total_score = 0
        self.hits.clear()
        self.phase = READY
        self.bull_streak = 0
        self.end_bonus()

    def bonus_active(self, now):
        return self.bonus_until is not None and now < self.bonus_until

    def bonus_left(self, now):
        """ボーナスの残り秒。ボーナス中でなければ 0。"""
        return max(0.0, self.bonus_until - now) if self.bonus_until is not None else 0.0

    def end_bonus(self):
        """ボーナスを即終了する。終了させたら True。"""
        was = self.bonus_until is not None
        self.bonus_start = self.bonus_until = None
        return was

    def update_bonus(self, now):
        """ボーナスの時間切れを処理する。今終わったら True。"""
        if self.bonus_until is not None and now >= self.bonus_until:
            return self.end_bonus()
        return False

    @property
    def ready(self):
        return self.phase == READY

    def register_hit(self, x, y, score, now, fever=False):
        """命中を記録して COOLDOWN に入る。戻り値: (倍率, ボーナスの変化 "start"/"extend"/None)

        ボーナス中の命中は2倍。fever でブルが BONUS_STREAK 回連続したらボーナス開始
        (ボーナス中なら終了時刻を now + BONUS_SEC に延長)。発動したブル自体は等倍。
        """
        mult = 2 if self.bonus_active(now) else 1
        self.total_score += score * mult
        self.hits.append(_make_hit(x, y, score, now, mult))
        self.phase = COOLDOWN
        self.cooldown_end = now + config.COOLDOWN_SEC

        self.bull_streak = self.bull_streak + 1 if (fever and score == 100) else 0
        event = None
        if self.bull_streak >= BONUS_STREAK:
            self.bull_streak = 0
            event = "extend" if self.bonus_active(now) else "start"
            if event == "start":
                self.bonus_start = now
            self.bonus_until = now + config.BONUS_SEC
        return mult, event

    def update(self, now):
        """COOLDOWN が終わったら READY に戻して True を返す。"""
        if self.phase == COOLDOWN and now >= self.cooldown_end:
            self.phase = READY
            return True
        return False

    def cooldown_progress(self, now):
        """COOLDOWN 中は 0→1、READY なら None。"""
        if self.phase != COOLDOWN:
            return None
        left = self.cooldown_end - now
        return min(1.0, max(0.0, 1.0 - left / config.COOLDOWN_SEC))
