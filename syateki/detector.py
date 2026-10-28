"""着弾検出とスコア判定 (純関数)。"""
from typing import NamedTuple, Tuple

import cv2
import numpy as np

from . import config

_mask_cache = {}


def warp_to_projector(cam_frame, corners, size=(config.PROJ_W, config.PROJ_H)):
    """カメラ画像を投影座標に透視変換する。(warped_bgr, ブラー済みグレー) を返す。"""
    m = cv2.getPerspectiveTransform(np.float32(corners), np.float32(
        [[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]))
    warped = cv2.warpPerspective(cam_frame, m, size)
    return warped, blur_gray(warped)


def blur_gray(bgr):
    return cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), config.BLUR_KSIZE, 0)


def score_for_distance(distance, radii=config.RADII):
    """中心からの距離 → 100 / 50 / 10、的の外なら 0。境界は内側扱い。"""
    r100, r50, r10 = radii
    if distance <= r100:
        return 100
    if distance <= r50:
        return 50
    if distance <= r10:
        return 10
    return 0


def score_for_point(x, y, center=config.TARGET_CENTER, radii=config.RADII):
    return score_for_distance(float(np.hypot(x - center[0], y - center[1])), radii)


def _target_mask(shape, center, radius):
    key = (shape, center, radius)
    m = _mask_cache.get(key)
    if m is None:
        m = np.zeros(shape, dtype=np.uint8)
        cv2.circle(m, center, radius + config.TARGET_MASK_MARGIN, 255, -1)
        _mask_cache[key] = m
    return m


class Hit(NamedTuple):
    """着弾。x, y は補正後 (int)、raw は補正前 (float)。method は "orange" / "lowest"。"""
    x: int
    y: int
    method: str
    orange_px: int
    raw: Tuple[float, float]


def _contact_point(contour, changed, mask_orange, blend):
    """候補輪郭から壁との接地点を求める。(x, y, method, 橙画素数) を返す。

    弾はオレンジの先端が壁に接し、青いボディは手前に飛び出している。カメラは
    的の斜め下にあるので、ボディは透視補正すると上にずれて写る。そこで
    1) 輪郭内の「差分あり かつ 橙」の重心、2) それが無ければ輪郭の重心と最下点を
    blend (0=重心, 1=最下点) で混ぜた点を使う。
    """
    x, y, w, h = cv2.boundingRect(contour)
    region = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(region, [contour - (x, y)], -1, 255, -1)
    orange = cv2.bitwise_and(changed[y:y + h, x:x + w], mask_orange[y:y + h, x:x + w])
    orange = cv2.bitwise_and(orange, region)
    ys, xs = np.nonzero(orange)
    n = int(len(xs))
    if n >= config.ORANGE_MIN_PIXELS:
        return x + float(xs.mean()), y + float(ys.mean()), "orange", n
    ry, rx = np.nonzero(region)
    cx, cy = x + float(rx.mean()), y + float(ry.mean())
    pts = contour.reshape(-1, 2)
    bottom = pts[pts[:, 1] == pts[:, 1].max()]
    bx, by = float(bottom[:, 0].mean()), float(bottom[0, 1])
    return cx + blend * (bx - cx), cy + blend * (by - cy), "lowest", n


def find_hit(bg_gray, frame_bgr, center=config.TARGET_CENTER, radius=config.R10,
             frame_gray=None, offset=None, blend=None):
    """背景グレー(ブラー済み)と現フレームBGRから着弾を探す。

    戻り値: (Hit または None, 弾センサーマスク)
    frame_gray は blur_gray(frame_bgr) を計算済みなら渡す。
    offset / blend を省くと config.HIT_OFFSET / config.FALLBACK_BLEND。
    """
    if offset is None:
        offset = config.HIT_OFFSET
    if blend is None:
        blend = config.FALLBACK_BLEND
    if frame_gray is None:
        frame_gray = blur_gray(frame_bgr)
    diff = cv2.absdiff(bg_gray, frame_gray)
    _, changed = cv2.threshold(diff, config.HIT_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    changed = cv2.dilate(changed, None, iterations=2)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(hsv, config.LOWER_BLUE, config.UPPER_BLUE)
    mask_orange = cv2.inRange(hsv, config.LOWER_ORANGE, config.UPPER_ORANGE)

    # 変化 × 弾の色 (影や人の動きを除く)
    candidate = cv2.bitwise_and(changed, cv2.bitwise_or(mask_blue, mask_orange))
    candidate = cv2.dilate(candidate, None, iterations=2)

    final = cv2.bitwise_and(candidate, _target_mask(candidate.shape, tuple(center), radius))
    contours, _ = cv2.findContours(final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if not (config.HIT_AREA_MIN < area < config.HIT_AREA_MAX):
            continue
        px, py, method, n = _contact_point(c, changed, mask_orange, blend)
        hx, hy = int(px + offset[0]), int(py + offset[1])
        if np.hypot(hx - center[0], hy - center[1]) <= radius:
            return Hit(hx, hy, method, n, (px, py)), candidate
    return None, candidate


def detect_hit(bg_gray, frame_bgr, center=config.TARGET_CENTER, radius=config.R10,
               frame_gray=None, offset=None, blend=None):
    """find_hit の座標だけ版。戻り値: (着弾座標 (x, y) または None, 弾センサーマスク)"""
    hit, mask = find_hit(bg_gray, frame_bgr, center, radius, frame_gray, offset, blend)
    return (None if hit is None else (hit.x, hit.y)), mask


class Settled(NamedTuple):
    """確定した着弾。reason は "settled" / "lost" / "timeout"。hit は lost+drop なら None。"""
    hit: "Hit | None"
    reason: str
    frames: int


class HitTracker:
    """候補が止まるまで追跡し、止まった位置で着弾を確定する。

    最初の候補フレームは飛行中・跳ね返り中のことが多いので使わない。
    連続 SETTLE_FRAMES フレームが SETTLE_PX 以内に収まったら最後の位置で確定。
    SETTLE_GRACE フレーム続けて見えなければ見失ったとみなし、SETTLE_LOST
    ("last": 最後に見えた位置 / "drop": 無効)。猶予中に再検出されたら追跡を続ける。
    SETTLE_TIMEOUT 秒を超えたらその時点の位置で確定。
    """

    def __init__(self, frames=None, px=None, timeout=None, lost=None, grace=None):
        self.frames = config.SETTLE_FRAMES if frames is None else frames
        self.px = config.SETTLE_PX if px is None else px
        self.timeout = config.SETTLE_TIMEOUT if timeout is None else timeout
        self.lost = config.SETTLE_LOST if lost is None else lost
        self.grace = config.SETTLE_GRACE if grace is None else grace
        self.reset()

    def reset(self):
        self.start = None   # 追跡開始時刻 (None = 追跡していない)
        self.last = None    # 最後に見えた Hit
        self.anchor = None  # 静止判定の基準点
        self.run = 0        # anchor から SETTLE_PX 以内に収まった連続フレーム数
        self.seen = 0       # 追跡開始からのフレーム数 (見えなかったフレームも数える)
        self.misses = 0     # 連続で見えなかったフレーム数

    @property
    def tracking(self):
        return self.start is not None

    def _done(self, hit, reason):
        out = Settled(hit, reason, self.seen)
        self.reset()
        return out

    def update(self, hit, now):
        """1フレーム分の候補 (Hit または None) を渡す。確定したら Settled、まだなら None。"""
        if not self.tracking:
            if hit is None:
                return None
            self.start = now
        elif hit is None:
            self.seen += 1
            self.misses += 1
            if self.misses >= self.grace:
                return self._done(self.last if self.lost == "last" else None, "lost")
            return None
        self.seen += 1
        self.misses = 0
        self.last = hit
        if self.anchor is not None and np.hypot(hit.x - self.anchor[0], hit.y - self.anchor[1]) <= self.px:
            self.run += 1
        else:
            self.anchor, self.run = (hit.x, hit.y), 1
        if self.run >= self.frames:
            return self._done(hit, "settled")
        if now - self.start >= self.timeout:
            return self._done(hit, "timeout")
        return None
