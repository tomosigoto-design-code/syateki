"""ArUco マーカーで投影範囲の四隅を求める (明るい会場でも効く自動キャリブ)。

投影画面に ArUco マーカー4個 (DICT_4X4_50, id 0〜3) を出し、カメラで見つけた
マーカー角と、投影座標での既知の角からホモグラフィを出す。それで投影画面の
四隅をカメラ座標へ写せば、corners.json と同じ形の四隅になる。
OpenCV 4.7+ / 5 の ArucoDetector と、それ以前の detectMarkers の両方に対応。
"""
import cv2
import numpy as np

from . import config

MARKER_IDS = (0, 1, 2, 3)  # 0:左上 1:右上 2:右下 3:左下
_screen_cache = {}


def _dictionary():
    aruco = cv2.aruco
    if hasattr(aruco, "getPredefinedDictionary"):
        return aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    return aruco.Dictionary_get(aruco.DICT_4X4_50)  # 古い 4.x


def _marker(marker_id, size):
    aruco = cv2.aruco
    if hasattr(aruco, "generateImageMarker"):
        return aruco.generateImageMarker(_dictionary(), marker_id, size)
    return aruco.drawMarker(_dictionary(), marker_id, size)  # 古い 4.x


def marker_layout(w=config.PROJ_W, h=config.PROJ_H, size=config.ARUCO_MARKER_PX,
                  margin=config.ARUCO_MARGIN_PX):
    """{id: 投影座標でのマーカー4角 (左上, 右上, 右下, 左下), float32 (4, 2)}"""
    xs = {0: margin, 1: w - margin - size, 2: w - margin - size, 3: margin}
    ys = {0: margin, 1: margin, 2: h - margin - size, 3: h - margin - size}
    return {i: np.float32([[xs[i], ys[i]], [xs[i] + size, ys[i]],
                           [xs[i] + size, ys[i] + size], [xs[i], ys[i] + size]])
            for i in MARKER_IDS}


def calib_screen(w=config.PROJ_W, h=config.PROJ_H):
    """キャリブ用の投影画面: 白地に4個のマーカー (周りの白がクワイエットゾーン)。"""
    key = (w, h)
    if key not in _screen_cache:
        img = np.full((h, w), 255, np.uint8)
        for i, quad in marker_layout(w, h).items():
            x, y = int(quad[0][0]), int(quad[0][1])
            img[y:y + config.ARUCO_MARKER_PX, x:x + config.ARUCO_MARKER_PX] = \
                _marker(i, config.ARUCO_MARKER_PX)
        _screen_cache[key] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return _screen_cache[key].copy()


def _detect_raw(gray):
    aruco = cv2.aruco
    if hasattr(aruco, "ArucoDetector"):
        det = aruco.ArucoDetector(_dictionary(), aruco.DetectorParameters())
        corners, ids, _ = det.detectMarkers(gray)
    else:  # 古い 4.x
        corners, ids, _ = aruco.detectMarkers(gray, _dictionary(),
                                              parameters=aruco.DetectorParameters_create())
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            if int(i) in MARKER_IDS:
                found[int(i)] = np.float32(c).reshape(4, 2)
    return found


def _variants(gray):
    """そのまま → CLAHE → ガンマ補正 (暗部を持ち上げる / 白飛び寄りを締める) の順に試す。"""
    yield gray
    yield cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    for g in (0.5, 2.0):
        lut = np.uint8(np.clip(((np.arange(256) / 255.0) ** g) * 255, 0, 255))
        yield cv2.LUT(gray, lut)


def detect_markers(gray):
    """カメラ画像 (グレー) から id 0〜3 のマーカーを探す。{id: (4, 2) のカメラ座標}"""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    best = {}
    for img in _variants(gray):
        found = _detect_raw(img)
        if len(found) > len(best):
            best = found
        if len(best) == len(MARKER_IDS):
            break
    return best


def corners_from_markers(found, w=config.PROJ_W, h=config.PROJ_H):
    """見つけたマーカー (2個以上) から投影画面の四隅をカメラ座標で返す。だめなら None。"""
    if len(found) < config.ARUCO_MIN_MARKERS:
        return None
    layout = marker_layout(w, h)
    ids = sorted(found)
    src = np.concatenate([layout[i] for i in ids])
    dst = np.concatenate([found[i] for i in ids])
    hmat, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if hmat is None:
        return None
    proj = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    cam = cv2.perspectiveTransform(proj, hmat).reshape(4, 2)
    if not np.all(np.isfinite(cam)) or cv2.contourArea(cam) < config.CALIB_MIN_AREA:
        return None
    return [[int(round(x)), int(round(y))] for x, y in cam]


class MarkerAverager:
    """複数フレームの検出結果をマーカーごとに平均して安定させる。"""

    def __init__(self):
        self.sums = {}
        self.counts = {}
        self.full_frames = 0  # 4個とも見えたフレーム数

    def add(self, found):
        for i, pts in found.items():
            self.sums[i] = self.sums.get(i, 0) + pts
            self.counts[i] = self.counts.get(i, 0) + 1
        if len(found) == len(MARKER_IDS):
            self.full_frames += 1

    def mean(self):
        return {i: self.sums[i] / self.counts[i] for i in self.sums}
