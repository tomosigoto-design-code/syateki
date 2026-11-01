"""無人運用のラウンド制 (config.AUTO_ROUND)。ATTRACT → PLAYING → RESULT → ATTRACT。

ATTRACT: 待機。最初の着弾で PLAYING (その1発目から数える)。
PLAYING: SHOTS_PER_ROUND 発、ROUND_TIMEOUT 秒、最後の着弾から IDLE_TIMEOUT 秒のどれかで終了。
RESULT : RESULT_SEC 秒結果を出す (着弾は無視、弾の回収時間)。その後 app が全部リセット。
"""
from . import config

ATTRACT, PLAYING, RESULT = "attract", "playing", "result"


def prize_for(score, tiers=None):
    """PRIZE_TIERS を上から見て、最初に score が届いた (label, kind)。無ければ (None, None)。"""
    for need, label, kind in (config.PRIZE_TIERS if tiers is None else tiers):
        if score >= need:
            return label, kind
    return None, None


def prize_hint(tiers=None, shots=None):
    """待機画面の呼び込み文 (例: "5発で300点以上 → 大きいお菓子")。最上位 (先頭) の tier を使う。

    tier が1つしかなければ (誰でももらえる) None。
    """
    tiers = config.PRIZE_TIERS if tiers is None else tiers
    shots = config.SHOTS_PER_ROUND if shots is None else shots
    if len(tiers) < 2:
        return None
    need, label, _ = tiers[0]
    return f"{shots}発で{need}点以上 → {label}"


class RoundState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = ATTRACT
        self.shots = 0
        self.started = None        # PLAYING に入った時刻
        self.last_hit = None       # 最後の着弾時刻
        self.result_started = None
        self.score = 0             # 確定したラウンド得点 (RESULT 中)
        self.prize = (None, None)  # (label, kind)
        self.end_reason = None     # "shots" / "round_timeout" / "idle_timeout"

    def accepts_hits(self):
        return self.state != RESULT

    def _finish(self, now, score, reason):
        self.state = RESULT
        self.result_started = now
        self.score = score
        self.prize = prize_for(score)
        self.end_reason = reason
        return "result"

    def on_hit(self, now, score):
        """着弾を登録した直後に呼ぶ (score はボーナス込みの合計)。終了したら "result"。"""
        if self.state == RESULT:
            return None
        if self.state == ATTRACT:
            self.state, self.started, self.shots = PLAYING, now, 0
        self.shots += 1
        self.last_hit = now
        if self.shots >= config.SHOTS_PER_ROUND:
            return self._finish(now, score, "shots")
        return None

    def deadline(self):
        """PLAYING の終了予定時刻 (早い方)。"""
        return min(self.started + config.ROUND_TIMEOUT, self.last_hit + config.IDLE_TIMEOUT)

    def update(self, now, score):
        """毎フレーム呼ぶ。"result" (タイムアウトで終了) / "reset" (結果表示が終わった) / None"""
        if self.state == PLAYING:
            if now - self.started >= config.ROUND_TIMEOUT:
                return self._finish(now, score, "round_timeout")
            if now - self.last_hit >= config.IDLE_TIMEOUT:
                return self._finish(now, score, "idle_timeout")
        elif self.state == RESULT and now - self.result_started >= config.RESULT_SEC:
            return "reset"
        return None

    def time_left(self, now):
        if self.state == PLAYING:
            return max(0.0, self.deadline() - now)
        if self.state == RESULT:
            return max(0.0, self.result_started + config.RESULT_SEC - now)
        return None

    def info(self, now):
        """render_projector に渡す round_info。"""
        return {"state": self.state, "shots": self.shots, "max_shots": config.SHOTS_PER_ROUND,
                "time_left": self.time_left(now), "prize_label": self.prize[0],
                "prize_kind": self.prize[1], "result_started": self.result_started,
                "hint": prize_hint()}

    def status(self, now):
        """監視画面用の1行 (ASCII のみ。cv2 の文字は日本語を描けない)。"""
        if self.state == ATTRACT:
            return "ROUND: WAITING (first hit starts)"
        if self.state == PLAYING:
            return f"ROUND: {self.shots}/{config.SHOTS_PER_ROUND} shots  {self.time_left(now):.0f}s"
        kind = (self.prize[1] or "none").upper()
        return f"RESULT {self.score}pt -> {kind} PRIZE  (next {self.time_left(now):.0f}s)"
