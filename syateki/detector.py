"""着弾検出とスコア判定 (純関数)。"""
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


def _contact_point(contour, changed, mask_orange):
    """候補輪郭から壁との接地点を求める。

    弾はオレンジの先端が壁に接し、青いボディは手前に飛び出している。カメラは
    的の斜め下にあるので、ボディは透視補正すると上にずれて写る。そこで
    1) 輪郭内の「差分あり かつ 橙」の重心、2) それが無ければ輪郭の最下点を使う。
    """
    x, y, w, h = cv2.boundingRect(contour)
    region = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(region, [contour - (x, y)], -1, 255, -1)
    orange = cv2.bitwise_and(changed[y:y + h, x:x + w], mask_orange[y:y + h, x:x + w])
    orange = cv2.bitwise_and(orange, region)
    ys, xs = np.nonzero(orange)
    if len(xs) >= config.ORANGE_MIN_PIXELS:
        return x + float(xs.mean()), y + float(ys.mean())
    pts = contour.reshape(-1, 2)
    bottom = pts[pts[:, 1] == pts[:, 1].max()]
    return float(bottom[:, 0].mean()), float(bottom[0, 1])


def detect_hit(bg_gray, frame_bgr, center=config.TARGET_CENTER, radius=config.R10,
               frame_gray=None, offset=None):
    """背景グレー(ブラー済み)と現フレームBGRから着弾を探す。

    戻り値: (着弾座標 (x, y) または None, 弾センサーマスク)
    frame_gray は blur_gray(frame_bgr) を計算済みなら渡す。
    """
    if offset is None:
        offset = config.HIT_OFFSET
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
        px, py = _contact_point(c, changed, mask_orange)
        hx, hy = int(px + offset[0]), int(py + offset[1])
        if np.hypot(hx - center[0], hy - center[1]) <= radius:
            return (hx, hy), candidate
    return None, candidate
