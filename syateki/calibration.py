"""四隅の並べ替え・フラッシュ式の四隅検出・corners.json の読み書き。"""
import json
import os

import cv2
import numpy as np

from . import config


def order_points(pts):
    """4点を (0:左上, 1:右上, 2:右下, 3:左下) に並べ替えて list で返す。"""
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = pts[:, 0] - pts[:, 1]
    rect[1] = pts[np.argmax(diff)]
    rect[3] = pts[np.argmin(diff)]
    return rect.tolist()


def detect_projection_corners(dark_gray, bright_gray):
    """暗フレームと白点灯フレーム(グレー)の差分から投影範囲の四隅を返す。

    見つからなければ None。戻り値は [[x, y], ...] (int, 左上から時計回り)。
    """
    diff = cv2.absdiff(bright_gray, dark_gray)
    _, thresh = cv2.threshold(diff, config.CALIB_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=3)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    if cv2.contourArea(best) <= config.CALIB_MIN_AREA:
        return None

    peri = cv2.arcLength(best, True)
    approx = cv2.approxPolyDP(best, 0.03 * peri, True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2)
    else:  # 4頂点にならなければ最小外接矩形
        quad = cv2.boxPoints(cv2.minAreaRect(best))
    return [[int(p[0]), int(p[1])] for p in order_points(quad)]


def load_corners(path=config.CONFIG_FILE, default=None):
    """corners.json を読む。無い・壊れているときは default のコピーを返す。"""
    if default is None:
        default = config.DEFAULT_CAMERA_CORNERS
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if len(data) == 4:
                corners = [[int(p[0]), int(p[1])] for p in data]
                print(f"★ 保存された角の設定を読み込みました: {corners}")
                return corners
        except Exception:
            pass
    return [list(p) for p in default]


def save_corners(corners, path=config.CONFIG_FILE):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump([[int(p[0]), int(p[1])] for p in corners], f)
    except Exception:
        pass
