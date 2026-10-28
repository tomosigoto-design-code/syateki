"""PC側カメラ監視・設定画面の描画（ネオンHUD調の操作パネル）"""
from __future__ import annotations

import cv2
import numpy as np

AA = cv2.LINE_AA
FONT = cv2.FONT_HERSHEY_DUPLEX
FONT_S = cv2.FONT_HERSHEY_SIMPLEX

# 呼び出し側がマウス判定に使うボタン座標 (x1, y1, x2, y2)
BUTTONS: dict[str, tuple[int, int, int, int]] = {
    "reset": (10, 10, 100, 42),
    "auto": (110, 10, 250, 42),
    "click4": (260, 10, 390, 42),
}

# ラベル, 塗り色, 縁色 (BGR)
_BUTTON_STYLE = {
    "reset": ("RESET", (40, 30, 170), (90, 90, 255)),
    "auto": ("AUTO ALIGN", (110, 120, 0), (230, 255, 60)),
    "click4": ("CLICK 4-PT", (0, 100, 200), (60, 180, 255)),
}

CYAN = (255, 230, 0)
GUIDE_HI = (0, 255, 255)
MASK_COLOR = (0, 165, 255)


def _fit_scale(text, font, scale, thick, max_w):
    """max_w に収まるようフォント倍率を縮める"""
    w = cv2.getTextSize(text, font, scale, thick)[0][0]
    if w > max_w > 0:
        scale *= max_w / w
    return max(scale, 0.25)


def _rounded_rect(img, x1, y1, x2, y2, r, color, thick=-1):
    """円+矩形で角丸矩形を描く（thick=-1 で塗り）"""
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    if thick < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, AA)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, AA)
        for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
            cv2.circle(img, (cx, cy), r, color, -1, AA)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thick, AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thick, AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thick, AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thick, AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thick, AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thick, AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thick, AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thick, AA)


def _text_center(img, text, cx, cy, font, scale, color, thick):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, (int(cx - tw / 2), int(cy + th / 2)), font, scale, color, thick, AA)


def _darken_roi(img, x1, y1, x2, y2, alpha):
    """矩形領域を黒で半透明に覆う（はみ出しはクリップ）"""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        roi = img[y1:y2, x1:x2]
        cv2.convertScaleAbs(roi, roi, 1.0 - alpha, 0)


def _draw_buttons(img, active_key):
    for key, (x1, y1, x2, y2) in BUTTONS.items():
        label, fill, edge = _BUTTON_STYLE[key]
        r = (y2 - y1) // 2
        _rounded_rect(img, x1, y1, x2, y2, r, fill, -1)
        if key == active_key:
            _rounded_rect(img, x1 - 2, y1 - 2, x2 + 2, y2 + 2, r + 2, (255, 255, 255), 2)
        else:
            _rounded_rect(img, x1, y1, x2, y2, r, edge, 1)
        s = _fit_scale(label, FONT, 0.5, 1, x2 - x1 - 2 * r + 8)
        _text_center(img, label, (x1 + x2) / 2, (y1 + y2) / 2, FONT, s, (255, 255, 255), 1)


def _draw_projection_area(img, corners):
    """投影範囲：薄い塗り＋発光線"""
    pts = np.array(corners, np.int32).reshape((-1, 1, 2))
    h, w = img.shape[:2]
    x, y, bw, bh = cv2.boundingRect(pts)
    pad = 6
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    if x2 > x1 and y2 > y1:
        roi = img[y1:y2, x1:x2]
        over = roi.copy()
        local = pts - np.array([[x1, y1]], np.int32)
        cv2.fillPoly(over, [local], CYAN, AA)
        cv2.addWeighted(over, 0.12, roi, 0.88, 0, roi)
        over = roi.copy()
        cv2.polylines(over, [local], True, CYAN, 7, AA)
        cv2.addWeighted(over, 0.35, roi, 0.65, 0, roi)
    cv2.polylines(img, [pts], True, CYAN, 2, AA)
    cv2.polylines(img, [pts], True, (255, 255, 220), 1, AA)


def _draw_corner(img, pt, color, label, selected, target, ui):
    h, w = img.shape[:2]
    x, y = int(pt[0]), int(pt[1])
    r = int(9 * ui)
    arm = int(16 * ui)
    if target:
        cv2.circle(img, (x, y), r + int(12 * ui), GUIDE_HI, 2, AA)
    if selected:
        cv2.circle(img, (x, y), r + int(7 * ui), (255, 255, 255), 2, AA)
        cv2.circle(img, (x, y), r + int(4 * ui), color, 2, AA)
    thick = 3 if selected else 2
    cv2.circle(img, (x, y), r, color, thick, AA)
    cv2.line(img, (x - arm, y), (x + arm, y), color, thick - 1 or 1, AA)
    cv2.line(img, (x, y - arm), (x, y + arm), color, thick - 1 or 1, AA)
    cv2.circle(img, (x, y), 2, (255, 255, 255), -1, AA)

    # ラベル：フレーム中心側へ寄せ、画面内にクリップ
    s = 0.45 * ui * (1.1 if selected else 1.0)
    (tw, th), base = cv2.getTextSize(label, FONT, s, 1)
    off = r + int(10 * ui)
    lx = x + off if x < w / 2 else x - off - tw
    ly = y + off + th if y < h / 2 else y - off
    lx = int(min(max(lx, 10), w - tw - 4))
    ly = int(min(max(ly, th + 4), h - base - 4))
    bar = max(3, int(4 * ui))
    _darken_roi(img, lx - 3, ly - th - 3, lx + tw + 3, ly + base + 2, 0.7)
    cv2.rectangle(img, (lx - 3 - bar, ly - th - 3), (lx - 4, ly + base + 2), color, -1)
    cv2.putText(img, label, (lx, ly), FONT, s, GUIDE_HI if target else (255, 255, 255), 1, AA)


def _draw_preview(img, src, x, y, pw, ph, label, color, ui):
    """枠・ラベル付きプレビュー。収まらない部分は切る"""
    h, w = img.shape[:2]
    if src.ndim == 2:
        m = cv2.resize(src, (pw, ph), interpolation=cv2.INTER_NEAREST)
        z = np.zeros_like(m)
        prev = cv2.merge([z, cv2.convertScaleAbs(m, alpha=color[1] / 255.0), cv2.convertScaleAbs(m, alpha=color[2] / 255.0)])
        prev = cv2.max(prev, 18)
    else:
        prev = cv2.resize(src, (pw, ph), interpolation=cv2.INTER_AREA)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + pw), min(h, y + ph)
    if x2 <= x1 or y2 <= y1:
        return
    img[y1:y2, x1:x2] = prev[y1 - y:y2 - y, x1 - x:x2 - x]
    cv2.rectangle(img, (x - 1, y - 1), (x + pw, y + ph), color, 1, AA)
    # 角のアクセント
    a = max(6, pw // 10)
    for cx, cy, dx, dy in ((x - 2, y - 2, 1, 1), (x + pw + 1, y - 2, -1, 1),
                           (x - 2, y + ph + 1, 1, -1), (x + pw + 1, y + ph + 1, -1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * a, cy), color, 2, AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * a), color, 2, AA)
    # ラベルタブ（プレビュー上辺の内側）
    s = _fit_scale(label, FONT, 0.4 * ui, 1, pw - 8)
    (tw, th), base = cv2.getTextSize(label, FONT, s, 1)
    cv2.rectangle(img, (x, y), (x + tw + 8, y + th + base + 6), color, -1)
    cv2.putText(img, label, (x + 4, y + th + 3), FONT, s, (20, 20, 20), 1, AA)


def render_monitor(cam_frame: np.ndarray, corners: list[list[int]], selected_idx: int,
                   click_step: int,
                   warped_bgr: np.ndarray | None,
                   dart_mask: np.ndarray | None,
                   corner_labels: list[str], corner_colors: list[tuple]) -> np.ndarray:
    """カメラ監視画面を描画した BGR 画像を返す（入力は書き換えない）

    click_step: -1 なら通常表示。0〜3 なら「次に [n] をクリック」ガイドを表示
    """
    img = cv2.cvtColor(cam_frame, cv2.COLOR_GRAY2BGR) if cam_frame.ndim == 2 else cam_frame.copy()
    h, w = img.shape[:2]
    ui = min(max(min(w / 640, h / 480), 0.6), 2.0)  # フレームサイズ基準の倍率
    m = int(10 * ui)
    clicking = 0 <= click_step < 4

    # 投影範囲
    if len(corners) == 4:
        _draw_projection_area(img, corners)

    # 右上プレビュー（ボタンと重なる幅ならボタンの下へ）
    pw = int(min(max(w * 0.24, 72), 360))
    ph = pw * 9 // 16
    px = w - pw - m
    py = m
    if px < BUTTONS["click4"][2] + m:
        py = BUTTONS["click4"][3] + m
    if warped_bgr is not None:
        _draw_preview(img, warped_bgr, px, py, pw, ph, "WARPED", CYAN, ui)
    if dart_mask is not None:
        _draw_preview(img, dart_mask, px, py + ph + m, pw, ph, "DART SENSOR", MASK_COLOR, ui)

    # 四隅ハンドル（プレビューより前面）
    for i, pt in enumerate(corners):
        _draw_corner(img, pt, corner_colors[i % len(corner_colors)], corner_labels[i % len(corner_labels)],
                     selected=(not clicking and i == selected_idx), target=(clicking and i == click_step), ui=ui)

    # 下部ガイド帯
    if clicking:
        lab = corner_labels[click_step % len(corner_labels)]
        line1 = f">> CLICK {lab} CORNER <<"
        line2 = f"step {click_step + 1}/4  -  click the projected corner on this view"
        c1, c2 = GUIDE_HI, (255, 255, 255)
    else:
        lab = corner_labels[selected_idx % len(corner_labels)] if corner_labels else ""
        line1 = "C: AUTO ALIGN / M: CLICK 4 / R: RESET / F: FULLSCREEN"
        line2 = f"Drag corner / Arrows,WASD: fine-tune {lab}"
        c1, c2 = (230, 255, 60), (210, 210, 210)
    max_w = w - 2 * m
    s1 = _fit_scale(line1, FONT, (0.7 if clicking else 0.5) * ui, 2 if clicking else 1, max_w)
    s2 = _fit_scale(line2, FONT_S, 0.45 * ui, 1, max_w)
    th1 = cv2.getTextSize(line1, FONT, s1, 1)[0][1]
    th2 = cv2.getTextSize(line2, FONT_S, s2, 1)[0][1]
    gap = int(8 * ui)
    bar_h = th1 + th2 + gap * 3
    _darken_roi(img, 0, h - bar_h, w, h, 0.65)
    if clicking:
        cv2.rectangle(img, (0, h - bar_h), (w - 1, h - 1), GUIDE_HI, 2)
    else:
        cv2.line(img, (0, h - bar_h), (w, h - bar_h), CYAN, 1, AA)
    cv2.putText(img, line1, (m, h - bar_h + gap + th1), FONT, s1, c1, 2 if clicking else 1, AA)
    cv2.putText(img, line2, (m, h - gap), FONT_S, s2, c2, 1, AA)

    # ボタンは最前面
    _draw_buttons(img, "click4" if clicking else None)
    return img
