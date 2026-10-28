"""ゲームの状態 (スコア・着弾履歴・READY/COOLDOWN)。演出は描画側が hits から計算する。"""
from dataclasses import dataclass, field

from . import config
from .render_projector import Hit

READY = "READY"
COOLDOWN = "COOLDOWN"


@dataclass
class GameState:
    total_score: int = 0
    hits: list = field(default_factory=list)  # list[Hit], 古い順
    phase: str = READY
    cooldown_end: float = 0.0

    def reset(self):
        self.total_score = 0
        self.hits.clear()
        self.phase = READY

    @property
    def ready(self):
        return self.phase == READY

    def register_hit(self, x, y, score, now):
        """命中を記録して COOLDOWN に入る。"""
        self.total_score += score
        self.hits.append(Hit(int(x), int(y), int(score), now))
        self.phase = COOLDOWN
        self.cooldown_end = now + config.COOLDOWN_SEC

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
