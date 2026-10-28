"""投影画面の描画(ネオン系アーケードHUD)。

状態は持たない。アニメーションは hits[].t と now だけから計算する。
静的な部分(背景・的・HUD枠・四隅マーカー)はサイズごとにキャッシュする。

カメラ誤検出対策:
- 中心から QUIET_R 以内は、命中演出を除いて完全に静止させる(回転・点滅は外側のみ、マスクで保証)。
- 命中演出は FX_END 秒以内にすべて消す(クールダウン 1.2 秒 + カメラ遅延より前)。
- 判定円内で動く演出の色は、弾の色(青 H95-135 / 橙 H5-25)を避ける。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Hit:
    x: int
    y: int
    score: int
    t: float  # time.time() での命中時刻


AA = cv2.LINE_AA
DUPLEX = cv2.FONT_HERSHEY_DUPLEX
TRIPLEX = cv2.FONT_HERSHEY_TRIPLEX

# BGR。静的な的リング用
CYAN = (255, 230, 40)
MAGENTA = (255, 50, 255)
RED = (60, 40, 255)
GOLD = (40, 210, 255)
WHITE = (255, 255, 255)
HUD = (200, 170, 40)
HUD_DIM = (110, 90, 25)
ORANGE = (0, 150, 255)
GREEN = (120, 255, 120)

# 命中演出用(判定円内で動く色。青・橙の色相を避ける)
FX_CYAN = (220, 255, 40)     # H≈85
FX_RED = (70, 50, 255)       # H≈177
FX_PINK = (200, 120, 255)    # H≈162
FX_WHITE = (235, 235, 255)
FX_COL = {100: FX_RED, 50: MAGENTA, 10: FX_CYAN}

# キャリブ用の四隅マーカー色(変更禁止)
CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 140, 0), (0, 255, 255)]

QUIET_R = 300        # この半径以内は命中演出以外を静止させる
FX_END = 0.8         # 命中演出が完全に消える時刻(秒)。0.85 未満
GLOW_DOWN = 4        # グロー計算の縮小率
COUNT_UP = 0.5       # スコアのカウントアップ時間(秒)
SCORE_DIGITS = 5

_static_cache: dict = {}


# ---------------------------------------------------------------- helpers
def _scale(col, a):
    a = max(0.0, min(1.0, a))
    return (int(col[0] * a), int(col[1] * a), int(col[2] * a))


def _glow(src, sigma=3.0, gain=1.6):
    """src を縮小→ぼかし→拡大したグロー画像を返す。"""
    h, w = src.shape[:2]
    small = cv2.resize(src, (w // GLOW_DOWN, h // GLOW_DOWN), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma)
    if gain != 1.0:
        small = cv2.convertScaleAbs(small, alpha=gain)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _text_center(img, text, cx, cy, font, scale, col, thick):
    """(cx, cy) を文字の中心にして描く。"""
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, (int(cx - tw / 2), int(cy + th / 2)), font, scale, col, thick, AA)


def _chamfer_poly(x0, y0, x1, y1, c):
    return np.array([[x0 + c, y0], [x1 - c, y0], [x1, y0 + c], [x1, y1 - c],
                     [x1 - c, y1], [x0 + c, y1], [x0, y1 - c], [x0, y0 + c]], np.int32)


def _panel_rect(w):
    pw, ph = 330, 60
    x0 = w // 2 - pw // 2
    return x0, 6, x0 + pw, 6 + ph


def _side_rects(w, h):
    """左(STATUS)・右(SHOTS)のサイドパネル。"""
    pw, ph = 190, 96
    y0 = h // 2 - ph // 2
    return (70, y0, 70 + pw, y0 + ph), (w - 70 - pw, y0, w - 70, y0 + ph)


# ---------------------------------------------------------------- static
def _build_static(w, h, center, radii):
    cx, cy = center
    r100, r50, r10 = radii

    # 背景: ほぼ黒 + グリッド
    bg = np.full((h, w, 3), (14, 9, 10), np.uint8)
    step = 40
    for x in range(cx % step, w, step):
        major = ((x - cx) // step) % 4 == 0
        cv2.line(bg, (x, 0), (x, h), (46, 30, 22) if major else (30, 20, 16), 1)
    for y in range(cy % step, h, step):
        major = ((y - cy) // step) % 4 == 0
        cv2.line(bg, (0, y), (w, y), (46, 30, 22) if major else (30, 20, 16), 1)

    # ビネット
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    vig = np.clip(1.15 - 0.55 * d, 0.25, 1.0)[..., None]
    bg = (bg.astype(np.float32) * vig).astype(np.uint8)

    # 的の内側を暗く半透明に塗る
    fill = bg.copy()
    cv2.circle(fill, center, r10, (40, 22, 6), -1, AA)
    cv2.circle(fill, center, r50, (46, 8, 40), -1, AA)
    cv2.circle(fill, center, r100, (12, 14, 70), -1, AA)
    bg = cv2.addWeighted(fill, 0.75, bg, 0.25, 0)
    for r, col in (((r50 + r10) // 2, (90, 70, 20)), ((r100 + r50) // 2, (90, 30, 80))):
        cv2.circle(bg, center, r, col, 1, AA)
    for (dx, dy) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        p0 = (cx + dx * (r100 + 8), cy + dy * (r100 + 8))
        p1 = (cx + dx * (r10 - 6), cy + dy * (r10 - 6))
        cv2.line(bg, p0, p1, (80, 60, 30), 1, AA)

    # 発光するもの(すべて静止なのでグローは一度だけ)
    glow_src = np.zeros_like(bg)
    rings = ((r10, CYAN), (r50, MAGENTA), (r100, RED))
    for r, col in rings:
        cv2.circle(glow_src, center, r, col, 10, AA)
    cv2.circle(glow_src, center, r100 - 10, GOLD, 4, AA)
    cv2.circle(glow_src, center, int(r100 * 0.45), GOLD, 3, AA)

    # 的の外周の目盛り(静止)
    n = 72
    ang = np.arange(n) * (2 * np.pi / n)
    long_ = (np.arange(n) % 6 == 0)
    r_in = np.where(long_, r10 + 8, r10 + 13).astype(np.float32)
    r_out = np.float32(r10 + 22)
    c, s = np.cos(ang), np.sin(ang)
    segs = np.round(np.stack([np.stack([cx + c * r_in, cy + s * r_in], 1),
                              np.stack([cx + c * r_out, cy + s * r_out], 1)], 1)).astype(np.int32)
    cv2.polylines(glow_src, list(segs[long_]), False, CYAN, 3, AA)

    # HUD パネル(スコア・左右)
    panels = [_chamfer_poly(*_panel_rect(w), 16)] + [_chamfer_poly(*r, 12) for r in _side_rects(w, h)]
    pfill = bg.copy()
    cv2.fillPoly(pfill, panels, (30, 16, 8), AA)
    bg = cv2.addWeighted(pfill, 0.85, bg, 0.15, 0)
    cv2.polylines(glow_src, panels, True, HUD, 4, AA)

    bg = cv2.add(bg, _glow(glow_src, sigma=3.0, gain=1.5))

    # 鋭い線(ネオン管: 色の太線 + 明るい芯)
    for r, col in rings:
        cv2.circle(bg, center, r, col, 4, AA)
        cv2.circle(bg, center, r, tuple(min(255, c // 2 + 128) for c in col), 1, AA)
    cv2.circle(bg, center, r100 - 10, GOLD, 2, AA)
    cv2.circle(bg, center, int(r100 * 0.45), GOLD, 1, AA)
    cv2.circle(bg, center, 4, WHITE, -1, AA)
    cv2.polylines(bg, list(segs[~long_]), False, (110, 95, 20), 1, AA)
    cv2.polylines(bg, list(segs[long_]), False, CYAN, 2, AA)
    cv2.polylines(bg, panels, True, HUD, 2, AA)

    # 画面外周の細い HUD 枠(四隅マーカーの所とスコアパネルの所は空ける)
    m = 14
    for a, b in (((90, m), (w // 2 - 200, m)), ((w // 2 + 200, m), (w - 90, m)),
                 ((90, h - m), (w - 90, h - m)), ((m, 90), (m, h - 90)), ((w - m, 90), (w - m, h - 90))):
        cv2.line(bg, a, b, HUD_DIM, 1, AA)
    for x in range(120, w - 110, 40):
        if abs(x - w // 2) > 200:
            cv2.line(bg, (x, m), (x, m + 5), HUD_DIM, 1, AA)
        cv2.line(bg, (x, h - m), (x, h - m - 5), HUD_DIM, 1, AA)

    # サイドパネルの見出し
    (lx0, ly0, _, _), (rx0, ry0, _, _) = _side_rects(w, h)
    cv2.putText(bg, "STATUS", (lx0 + 18, ly0 + 22), DUPLEX, 0.45, HUD, 1, AA)
    cv2.putText(bg, "SHOTS", (rx0 + 18, ry0 + 22), DUPLEX, 0.45, HUD, 1, AA)
    x0, y0, _, _ = _panel_rect(w)
    cv2.putText(bg, "SCORE", (x0 + 22, y0 + 20), DUPLEX, 0.45, HUD, 1, AA)

    # 得点ラベル(各帯の上下)
    labels = (("10", (r50 + r10) / 2, CYAN), ("50", (r100 + r50) / 2, MAGENTA))
    lab_glow = np.zeros_like(bg)
    for text, rr, col in labels:
        for sgn in (-1, 1):
            _text_center(lab_glow, text, cx, cy + sgn * rr, DUPLEX, 1.0, col, 5)
    bg = cv2.add(bg, _glow(lab_glow, sigma=2.0, gain=1.0))
    for text, rr, col in labels:
        for sgn in (-1, 1):
            (tw, th), _ = cv2.getTextSize(text, DUPLEX, 1.0, 2)
            y = int(cy + sgn * rr)
            tag = _chamfer_poly(cx - tw // 2 - 10, y - th // 2 - 8, cx + tw // 2 + 10, y + th // 2 + 8, 6)
            cv2.fillPoly(bg, [tag], (10, 6, 6), AA)
            cv2.polylines(bg, [tag], True, _scale(col, 0.45), 1, AA)
            _text_center(bg, text, cx, y, DUPLEX, 1.0, col, 2)

    _draw_corners(bg, w, h)

    _text_center(bg, "C: AUTO ALIGN  /  RIGHT CLICK: RESET  /  F: FULLSCREEN",
                 w // 2, h - 26, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 110, 100), 1)

    # 判定円の外だけ通すマスク(動く HUD 要素用)
    outer = np.full((h, w, 3), 255, np.uint8)
    cv2.circle(outer, center, QUIET_R, (0, 0, 0), -1)
    return bg, outer


def _draw_corners(img, w, h):
    inset, arm = 40, 34
    corners = [((inset, inset), (1, 1)), ((w - inset, inset), (-1, 1)),
               ((w - inset, h - inset), (-1, -1)), ((inset, h - inset), (1, -1))]
    for i, ((x, y), (sx, sy)) in enumerate(corners):
        col = CORNER_COLORS[i]
        pts = np.array([[x + sx * arm, y], [x, y], [x, y + sy * arm]], np.int32)
        cv2.polylines(img, [pts], False, col, 4, AA)
        cv2.circle(img, (x, y), 5, col, -1, AA)
        _text_center(img, str(i + 1), x + sx * 20, y + sy * 20, DUPLEX, 0.6, col, 2)


def _get_static(w, h, center, radii):
    key = (w, h, center, radii)
    v = _static_cache.get(key)
    if v is None:
        if len(_static_cache) > 8:
            _static_cache.clear()
        v = _build_static(w, h, center, radii)
        _static_cache[key] = v
    return v


# ---------------------------------------------------------------- outer (判定円の外で動くもの)
def _draw_outer_motion(lay, center, now, cooldown_progress):
    """判定円の外側の回転リング / リロードゲージ。"""
    cx, cy = center
    rt = QUIET_R + 12
    # 回転する目盛り
    n = 90
    ang = now * 0.25 + np.arange(n) * (2 * np.pi / n)
    long_ = (np.arange(n) % 5 == 0)
    r_in = np.where(long_, rt - 4, rt).astype(np.float32)
    c, s = np.cos(ang), np.sin(ang)
    segs = np.round(np.stack([np.stack([cx + c * r_in, cy + s * r_in], 1),
                              np.stack([cx + c * (rt + 8), cy + s * (rt + 8)], 1)], 1)).astype(np.int32)
    cv2.polylines(lay, list(segs[~long_]), False, (90, 75, 20), 1, AA)
    cv2.polylines(lay, list(segs[long_]), False, (200, 180, 40), 2, AA)

    rg = rt + 18
    if cooldown_progress is None:
        base = -math.degrees(now * 0.4)
        for k in range(4):
            a0 = base + k * 90
            cv2.ellipse(lay, center, (rg, rg), 0, a0, a0 + 35, (140, 115, 30), 2, AA)
    else:
        p = max(0.0, min(1.0, cooldown_progress))
        cv2.ellipse(lay, center, (rg, rg), 0, -90, 270, (40, 40, 60), 2, AA)
        if p > 0:
            cv2.ellipse(lay, center, (rg, rg), 0, -90, -90 + 360 * p, ORANGE, 3, AA)


def _draw_score(lay, w, total_score, hits, now):
    x0, y0, x1, y1 = _panel_rect(w)
    shown = total_score
    bump = 0.0
    if hits:
        last = hits[-1]
        age = now - last.t
        if age >= 0:
            p = min(1.0, age / COUNT_UP)
            p = 1 - (1 - p) ** 2
            shown = int(round(total_score - last.score * (1 - p)))
            bump = max(0.0, 1 - age / 0.3) ** 2
    shown = max(0, shown)

    # 等幅の桁セルに1文字ずつ置く。先頭のゼロは暗く。
    sc = 1.35 * (1 + 0.22 * bump)
    cell = max(cv2.getTextSize(d, DUPLEX, sc, 3)[0][0] for d in "0123456789") + 4
    s = str(shown).rjust(SCORE_DIGITS, "0")
    nlead = len(s) - len(str(shown)) if shown > 0 else len(s) - 1
    cx = (x0 + x1) // 2 + 16
    cy = (y0 + y1) // 2 + 2
    left = cx - cell * len(s) / 2
    hot = (235, 255, 250) if bump > 0 else (190, 235, 225)
    for i, d in enumerate(s):
        dc = left + cell * (i + 0.5)
        if i < nlead:
            _text_center(lay, d, dc, cy, DUPLEX, sc, (60, 45, 15), 2)
        else:
            _text_center(lay, d, dc, cy, DUPLEX, sc, _scale(CYAN, 0.6), 4)
            _text_center(lay, d, dc, cy, DUPLEX, sc, hot, 1)


def _draw_side(lay, w, h, hits, cooldown_progress, now):
    (lx0, ly0, lx1, ly1), (rx0, ry0, rx1, ry1) = _side_rects(w, h)
    # STATUS: READY 点滅 / RELOAD ゲージ
    mx, my = (lx0 + lx1) // 2, ly0 + 56
    if cooldown_progress is None:
        a = 0.5 + 0.4 * (0.5 + 0.5 * math.sin(now * 4.0))
        _text_center(lay, "READY", mx, my, DUPLEX, 1.0, _scale(GREEN, a), 2)
    else:
        p = max(0.0, min(1.0, cooldown_progress))
        _text_center(lay, "RELOAD", mx, my - 6, DUPLEX, 0.8, ORANGE, 2)
        bx0, bx1, by = lx0 + 20, lx1 - 20, my + 20
        cv2.rectangle(lay, (bx0, by - 4), (bx1, by + 4), (50, 50, 70), 1, AA)
        if p > 0:
            cv2.rectangle(lay, (bx0 + 2, by - 2), (bx0 + 2 + int((bx1 - bx0 - 4) * p), by + 2), ORANGE, -1)
    # SHOTS: 発数と直近の得点
    mx = (rx0 + rx1) // 2
    _text_center(lay, f"{len(hits):02d}", mx - 34, ry0 + 58, DUPLEX, 1.2, _scale(CYAN, 0.6), 4)
    _text_center(lay, f"{len(hits):02d}", mx - 34, ry0 + 58, DUPLEX, 1.2, (190, 235, 225), 1)
    if hits:
        last = hits[-1].score
        col = {100: GOLD, 50: MAGENTA}.get(last, CYAN)
        _text_center(lay, f"+{last}", mx + 44, ry0 + 58, DUPLEX, 0.8, col, 2)


# ---------------------------------------------------------------- inner (命中演出・着弾痕)
def _draw_hit_marks(lay, hits):
    """直近6発。明るさは新しさの順位だけで決める(次の命中まで静止)。"""
    recent = hits[-6:]
    n = len(recent)
    for i, hit in enumerate(recent):
        a = 0.3 + 0.7 * (i + 1) / n
        col = FX_COL.get(hit.score, FX_CYAN)
        p = (int(hit.x), int(hit.y))
        cv2.circle(lay, p, 11, _scale(col, a), 2, AA)
        cv2.circle(lay, p, 3, _scale(WHITE, a), -1, AA)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            cv2.line(lay, (p[0] + dx * 15, p[1] + dy * 15), (p[0] + dx * 21, p[1] + dy * 21),
                     _scale(col, a), 2, AA)


def _draw_hit_fx(shade, lay, hit: Hit, age: float):
    x, y = int(hit.x), int(hit.y)
    col = FX_COL.get(hit.score, FX_CYAN)
    big = hit.score == 100

    # 着弾の閃光
    if age < 0.1:
        a = 1 - age / 0.1
        cv2.circle(lay, (x, y), int(16 + 26 * (1 - a)), _scale(FX_WHITE, a), -1, AA)

    # 衝撃波リング(3重)。0.7 秒で消える
    rmax = 190 if big else (140 if hit.score == 50 else 110)
    for k, delay in enumerate((0.0, 0.07, 0.14)):
        t = (age - delay) / (0.7 - delay)
        if 0 < t < 1:
            e = 1 - (1 - t) ** 3
            r = int(10 + e * rmax * (1 - 0.18 * k))
            a = (1 - t) ** 1.3
            thick = max(1, int(round((6 if k == 0 else 3) * (1 - t) + 1)))
            c = (FX_WHITE, 0.6) if k == 1 else (col, 1.0)
            cv2.circle(lay, (x, y), r, _scale(c[0], a * c[1]), thick, AA)

    # パーティクル(ヒットごとに固定シード)。寿命は最大 0.75 秒
    rng = np.random.default_rng(int(hit.t * 1000) & 0xFFFFFFFF)
    n = 64 if big else (42 if hit.score == 50 else 28)
    ang = rng.uniform(0, 2 * np.pi, n)
    spd = rng.uniform(220, 640 if big else 480, n)
    plife = rng.uniform(0.4, 0.75, n)
    kind = rng.integers(0, 3, n)
    k_drag, g = 3.2, 300.0

    def pos(t):
        f = (1 - np.exp(-k_drag * t)) / k_drag
        return x + np.cos(ang) * spd * f, y + np.sin(ang) * spd * f + 0.5 * g * t * t

    ta = np.minimum(age, plife)
    xa, ya = pos(ta)
    xb, yb = pos(np.maximum(ta - 0.035, 0))
    fade = np.clip(1 - age / plife, 0, 1) ** 1.4
    palette = [col, FX_WHITE, FX_PINK if big else col]
    for i in np.nonzero(fade > 0.02)[0]:
        cv2.line(lay, (int(xb[i]), int(yb[i])), (int(xa[i]), int(ya[i])),
                 _scale(palette[kind[i]], fade[i]), 2, AA)

    # 得点ポップアップ
    rise = 1 - (1 - min(1.0, age / 0.6)) ** 2
    py = int(y - 46 - 50 * rise)
    pop = 1 + 0.5 * max(0.0, 1 - age / 0.12) ** 2
    a = 1.0 if age < 0.45 else max(0.0, 1 - (age - 0.45) / (FX_END - 0.45))
    sc = (1.25 if big else 1.0) * pop
    txt = f"+{hit.score}"
    # 縁取り: shade に描いた分だけ下地を暗くする(フェードに追従)
    _text_center(shade, txt, x, py, DUPLEX, sc, _scale(WHITE, a), 12)
    _text_center(lay, txt, x, py, DUPLEX, sc, _scale(col, a * 0.85), 5)
    _text_center(lay, txt, x, py, DUPLEX, sc, _scale(FX_WHITE, a * 0.9), 1)


def _draw_bullseye_banner(shade, lay, w, h, center, age):
    """100点の大きな BULLSEYE。shade で帯と縁を暗くし、lay に文字を描く。"""
    slam = max(0.0, 1 - age / 0.12)
    sc = 2.5 * (1 + 0.35 * slam ** 2)
    a = 1.0 if age < 0.5 else max(0.0, 1 - (age - 0.5) / (FX_END - 0.5))
    cy = min(h - 120, center[1] + 150)
    y0, y1 = max(0, cy - 50), min(h, cy + 50)
    cv2.rectangle(shade, (0, y0), (w, y1), _scale((190, 190, 190), a), -1)
    bw = int(w * 0.5 * min(1.0, age / 0.1 + 0.25))
    for yy in (y0, y1):
        cv2.line(lay, (w // 2 - bw, yy), (w // 2 + bw, yy), _scale(FX_RED, a), 2, AA)
    txt = "BULLSEYE"
    off = int(3 + 10 * slam)
    _text_center(shade, txt, w // 2, cy, TRIPLEX, sc, _scale(WHITE, a), 14)
    _text_center(lay, txt, w // 2 - off, cy, TRIPLEX, sc, _scale((120, 0, 200), a * 0.6), 6)
    _text_center(lay, txt, w // 2 + off, cy, TRIPLEX, sc, _scale(FX_CYAN, a * 0.35), 6)
    _text_center(lay, txt, w // 2, cy, TRIPLEX, sc, _scale((50, 30, 230), a * 0.8), 6)
    _text_center(lay, txt, w // 2, cy, TRIPLEX, sc, _scale((190, 180, 255), a * 0.9), 1)


# ---------------------------------------------------------------- main
def render_projector(w: int, h: int, center: tuple[int, int], radii: tuple[int, int, int],
                     total_score: int, hits: list[Hit],
                     cooldown_progress: float | None,
                     calibrating: bool, now: float) -> np.ndarray:
    """投影画面を1フレーム描いて BGR uint8 (h, w, 3) を返す。

    radii は (r100, r50, r10)。hits は古い順の全履歴。
    cooldown_progress はクールダウン中 0→1、READY なら None。
    calibrating=True のときは真っ白(カメラの投影範囲検出用)。
    """
    if calibrating:
        img = np.full((h, w, 3), 255, np.uint8)
        _text_center(img, "CALIBRATING", w // 2, h // 2, DUPLEX, 0.9, (0, 0, 0), 2)
        return img

    center = (int(center[0]), int(center[1]))
    radii = tuple(int(r) for r in radii)
    base, outer_mask = _get_static(w, h, center, radii)
    frame = base.copy()

    # 判定円の外で動くもの(回転リング・スコア・STATUS)。マスクで円内には一切出さない
    outer = np.zeros_like(frame)
    _draw_outer_motion(outer, center, now, cooldown_progress)
    x0, y0, x1, y1 = _panel_rect(w)
    outer[y0 + 3:y1 - 2, x0 + 10:x1 - 10] = 0
    _draw_score(outer, w, total_score, hits, now)
    _draw_side(outer, w, h, hits, cooldown_progress, now)
    cv2.add(outer, _glow(outer, sigma=2.5, gain=1.6), dst=outer)
    cv2.bitwise_and(outer, outer_mask, dst=outer)
    cv2.add(frame, outer, dst=frame)

    # 着弾痕と命中演出(FX_END 秒で完全に消える)
    inner = np.zeros_like(frame)
    _draw_hit_marks(inner, hits)
    flash_white = flash_red = 0.0
    active = [(hit, now - hit.t) for hit in hits[-8:] if 0 <= now - hit.t < FX_END]
    if active:
        shade = np.zeros_like(frame)
        for hit, age in active:
            _draw_hit_fx(shade, inner, hit, age)
            if hit.score == 100:
                _draw_bullseye_banner(shade, inner, w, h, center, age)
                flash_white = max(flash_white, 1 - age / 0.08)
                flash_red = max(flash_red, 1 - age / 0.3)
        cv2.subtract(frame, shade, dst=frame)
    cv2.add(frame, inner, dst=frame)
    cv2.add(frame, _glow(inner, sigma=2.5, gain=1.6), dst=frame)

    # 100点の画面フラッシュ(白→赤、0.3 秒で終わる)
    if flash_white > 0 or flash_red > 0:
        fw = 150 * max(0.0, flash_white)
        fr = 80 * max(0.0, flash_red)
        cv2.add(frame, (fw + fr * 0.1, fw + fr * 0.1, fw + fr, 0), dst=frame)
    return frame
