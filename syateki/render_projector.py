"""投影画面の描画(ダーツボード + 中継風スコアバー)。

状態は持たない。アニメーションは hits[].t と now だけから計算する。
静的な部分(背景・ボード・スコアバーの枠・四隅マーカー)はサイズごとにキャッシュする。

カメラ誤検出対策:
- 中心から QUIET_R 以内は、命中演出を除いて完全に静止させる(外で動くのはスコアバーとブルの祝祭演出だけ、マスクで保証)。
- 命中演出は FX_END 秒以内にすべて消す(クールダウン 1.2 秒 + カメラ遅延より前)。
- 判定円内で動く色は白・黒・グレーと、ボード自身の色を白へ寄せたものだけ(弾の青・橙を出さない)。
- 全画面フラッシュはしない(カメラの露出が揺れるため)。
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
SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX

# BGR
BG = (30, 31, 33)             # チャコール
BOARD_BLACK = (24, 24, 24)
BOARD_CREAM = (190, 222, 234)
BOARD_RED = (48, 42, 196)
BOARD_GREEN = (70, 132, 32)
WIRE = (172, 174, 176)
WHITE = (255, 255, 255)
INK = (12, 12, 12)
BAR_DARK = (22, 22, 23)
BAR_MID = (40, 40, 42)
LABEL = (150, 150, 152)
READY_GREEN = (90, 190, 70)

# キャリブ用の四隅マーカー色(変更禁止)
CORNER_COLORS = [(0, 0, 255), (0, 255, 0), (255, 140, 0), (0, 255, 255)]

QUIET_R = 300        # この半径以内は命中演出以外を静止させる
FX_END = 0.8         # 命中演出が完全に消える時刻(秒)。0.85 未満
COUNT_UP = 0.35      # スコアのカウントアップ(判定円の外)
SS = 2               # ボードを描くときの超解像倍率

_static_cache: dict = {}


# ---------------------------------------------------------------- geometry
def _board_rings(radii):
    """的の各リングの半径。色の境界を得点境界 (r100, r50, r10) に合わせる。"""
    r100, r50, r10 = radii
    return {
        "bull": r100,                        # 赤(100点)
        "middle": r50,                       # 緑(50点)
        "outer": r10,                        # クリーム(10点)
        "edge": r10 + round(r10 * 0.1),      # 黒い外周リングの外側
    }


def _text_center(img, text, cx, cy, font, scale, col, thick):
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    cv2.putText(img, text, (int(round(cx - tw / 2)), int(round(cy + th / 2))), font, scale, col, thick, AA)


def _bar_rect(w, h, center, radii):
    """スコアバー (x0, y0, x1, y1)。ボードの左に置く。"""
    edge = _board_rings(radii)["edge"]
    x0 = 72
    x1 = max(x0 + 150, min(x0 + 214, center[0] - max(edge, QUIET_R) - 28))
    y0 = h // 2 - 118
    return x0, y0, x1, y0 + 156


# ---------------------------------------------------------------- static
def _draw_board(size, cx, cy, radii):
    """3重リングの的(外=クリーム、中=緑、ブル=赤に黒縁)を SS 倍で描いて縮小した (img, alpha) を返す。"""
    R = _board_rings(radii)
    S = SS
    big = np.zeros((size * S, size * S, 3), np.uint8)
    alpha = np.zeros((size * S, size * S), np.uint8)
    c = (cx * S, cy * S)

    cv2.circle(big, c, R["edge"] * S, BOARD_BLACK, -1, AA)
    cv2.circle(alpha, c, R["edge"] * S, 255, -1, AA)
    cv2.circle(big, c, R["outer"] * S, BOARD_CREAM, -1, AA)
    cv2.circle(big, c, R["middle"] * S, BOARD_GREEN, -1, AA)
    cv2.circle(big, c, R["bull"] * S, BOARD_RED, -1, AA)

    # 境界線: ブルは黒縁、中間の外周は黒の細線、外周は銀のワイヤー
    cv2.circle(big, c, R["bull"] * S, BOARD_BLACK, 3 * S, AA)
    cv2.circle(big, c, R["middle"] * S, BOARD_BLACK, 2 * S, AA)
    cv2.circle(big, c, R["outer"] * S, WIRE, S, AA)
    cv2.circle(big, c, (R["edge"] - 2) * S, (58, 58, 60), S, AA)

    img = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (size, size), interpolation=cv2.INTER_AREA)
    return img, alpha



def _band_icon(img, cx, cy, zone):
    """得点の説明用の色見本(的と同じ色の小さな円)。"""
    col = (BOARD_RED, BOARD_GREEN, BOARD_CREAM)[zone]
    cv2.circle(img, (cx, cy), 8, col, -1, AA)
    cv2.circle(img, (cx, cy), 8, (90, 90, 92), 1, AA)


def _build_static(w, h, center, radii):
    cx, cy = center
    R = _board_rings(radii)
    bg = np.empty((h, w, 3), np.uint8)
    bg[:] = BG

    # ごく弱い影(壁に掛かったボード)。上から光が当たる想定で少し下へ
    sh = np.zeros((h, w), np.float32)
    cv2.circle(sh, (cx, cy + 9), R["edge"] + 2, 1.0, -1, AA)
    sh = cv2.GaussianBlur(sh, (0, 0), 14)
    bg = (bg.astype(np.float32) * (1 - 0.5 * sh[..., None])).astype(np.uint8)

    # ボード
    half = R["edge"] + 3
    x0, y0 = cx - half, cy - half
    img, alpha = _draw_board(2 * half, half, half, radii)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = min(w, x0 + 2 * half), min(h, y0 + 2 * half)
    if dx1 > dx0 and dy1 > dy0:
        sub = (slice(sy0, sy0 + dy1 - dy0), slice(sx0, sx0 + dx1 - dx0))
        a = alpha[sub].astype(np.float32)[..., None] / 255
        roi = bg[dy0:dy1, dx0:dx1]
        bg[dy0:dy1, dx0:dx1] = (img[sub] * a + roi * (1 - a) + 0.5).astype(np.uint8)

    # スコアバー(枠と見出しだけ。数字は毎フレーム)
    bx0, by0, bx1, by1 = _bar_rect(w, h, center, radii)
    cv2.rectangle(bg, (bx0, by0), (bx1, by0 + 80), BAR_DARK, -1)
    cv2.rectangle(bg, (bx0, by0), (bx0 + 3, by0 + 80), BOARD_RED, -1)
    cv2.rectangle(bg, (bx0, by0 + 80), (bx1, by0 + 126), BAR_MID, -1)
    cv2.rectangle(bg, (bx0, by0 + 126), (bx1, by1), BAR_DARK, -1)
    cv2.putText(bg, "SCORE", (bx0 + 16, by0 + 22), SIMPLEX, 0.4, LABEL, 1, AA)
    mid = (bx0 + bx1) // 2
    cv2.putText(bg, "SHOTS", (bx0 + 16, by0 + 97), SIMPLEX, 0.36, LABEL, 1, AA)
    cv2.putText(bg, "LAST", (mid + 8, by0 + 97), SIMPLEX, 0.36, LABEL, 1, AA)
    cv2.line(bg, (mid, by0 + 88), (mid, by0 + 118), (58, 58, 60), 1)

    # 得点の説明(色見本 + 点数)
    ky = by1 + 34
    step = (bx1 - bx0) // 3
    for i, (zone, pts) in enumerate(((0, "100"), (1, "50"), (2, "10"))):
        kx = bx0 + 16 + i * step
        _band_icon(bg, kx + 9, ky, zone)
        cv2.putText(bg, pts, (kx + 24, ky + 6), SIMPLEX, 0.45, (200, 200, 200), 1, AA)

    _draw_corners(bg, w, h)
    _text_center(bg, "C: AUTO ALIGN   /   RIGHT CLICK: RESET   /   F: FULLSCREEN",
                 w // 2, h - 18, SIMPLEX, 0.38, (88, 89, 92), 1)

    # 判定円の内側 (QUIET_R) のマスク。スコアバーの描画がはみ出しても戻す
    quiet = np.zeros((h, w), np.uint8)
    cv2.circle(quiet, center, QUIET_R, 255, -1)

    # 得点帯のマスク(命中時に一瞬だけ明るくする)。ボードの外接正方形で持つ
    r10 = radii[2]
    bx = (max(0, cx - r10 - 1), max(0, cy - r10 - 1), min(w, cx + r10 + 2), min(h, cy + r10 + 2))
    zones = {}
    prev = 0
    for score, r in ((100, radii[0]), (50, radii[1]), (10, radii[2])):
        m = np.zeros((bx[3] - bx[1], bx[2] - bx[0]), np.uint8)
        cv2.circle(m, (cx - bx[0], cy - bx[1]), r, 255, -1)
        if prev:
            cv2.circle(m, (cx - bx[0], cy - bx[1]), prev, 0, -1)
        zones[score] = m
        prev = r
    return {"bg": bg, "quiet": quiet, "zone_box": bx, "zones": zones}


def _draw_corners(img, w, h):
    """四隅のキャリブ用マーカー(L字 + 番号)。色は固定。"""
    inset, arm = 40, 26
    corners = [((inset, inset), (1, 1)), ((w - inset, inset), (-1, 1)),
               ((w - inset, h - inset), (-1, -1)), ((inset, h - inset), (1, -1))]
    for i, ((x, y), (sx, sy)) in enumerate(corners):
        col = CORNER_COLORS[i]
        pts = np.array([[x + sx * arm, y], [x, y], [x, y + sy * arm]], np.int32)
        cv2.polylines(img, [pts], False, col, 2, AA)
        _text_center(img, str(i + 1), x + sx * 13, y + sy * 13, SIMPLEX, 0.4, col, 1)


def _get_static(w, h, center, radii):
    key = (w, h, center, radii)
    v = _static_cache.get(key)
    if v is None:
        if len(_static_cache) > 8:
            _static_cache.clear()
        v = _build_static(w, h, center, radii)
        _static_cache[key] = v
    return v


# ---------------------------------------------------------------- score bar (判定円の外)
def _draw_bar(frame, w, h, center, radii, total_score, hits, cooldown_progress, now):
    bx0, by0, bx1, by1 = _bar_rect(w, h, center, radii)
    shown = total_score
    col = WHITE
    if hits:
        age = now - hits[-1].t
        dur = COUNT_UP * 2 if hits[-1].score == 100 else COUNT_UP
        if 0 <= age < dur:
            p = 1 - (1 - age / dur) ** 2
            shown = int(round(total_score - hits[-1].score * (1 - p)))
        if hits[-1].score == 100 and 0 <= age < BULL_END:
            k = 1.0 if age < 1.2 else max(0.0, 1 - (age - 1.2) / (BULL_END - 1.2))
            col = tuple(int(g * k + 255 * (1 - k)) for g in GOLD)
    txt = str(max(0, shown))
    (tw, _), _ = cv2.getTextSize(txt, DUPLEX, 1.55, 2)
    cv2.putText(frame, txt, (bx1 - 16 - tw, by0 + 66), DUPLEX, 1.55, col, 2, AA)

    mid = (bx0 + bx1) // 2
    cv2.putText(frame, str(len(hits)), (bx0 + 16, by0 + 118), DUPLEX, 0.7, WHITE, 1, AA)
    if hits:
        cv2.putText(frame, str(hits[-1].score), (mid + 8, by0 + 118), DUPLEX, 0.7, WHITE, 1, AA)

    sy = by0 + 141
    if cooldown_progress is None:
        cv2.circle(frame, (bx0 + 20, sy), 4, READY_GREEN, -1, AA)
        cv2.putText(frame, "READY", (bx0 + 32, sy + 5), SIMPLEX, 0.42, (225, 225, 225), 1, AA)
    else:
        p = max(0.0, min(1.0, cooldown_progress))
        cv2.putText(frame, "RELOAD", (bx0 + 16, sy + 5), SIMPLEX, 0.42, LABEL, 1, AA)
        gx0, gx1 = bx0 + 82, bx1 - 16
        cv2.line(frame, (gx0, sy), (gx1, sy), (70, 70, 72), 2)
        if p > 0:
            cv2.line(frame, (gx0, sy), (gx0 + int((gx1 - gx0) * p), sy), (215, 215, 215), 2)


# ---------------------------------------------------------------- inner (着弾跡・命中演出)
def _draw_hit_marks(frame, hits):
    """直近6発の刺さり跡。明るさは新しさの順位だけで決める(次の命中まで静止)。"""
    recent = hits[-6:]
    n = len(recent)
    for i, hit in enumerate(recent):
        v = int(120 + 135 * (i + 1) / n)
        x, y = int(hit.x), int(hit.y)
        cv2.line(frame, (x + 2, y + 2), (x + 7, y + 9), INK, 3, AA)   # 短い影
        cv2.circle(frame, (x, y), 4, INK, -1, AA)
        cv2.circle(frame, (x, y), 3, (v, v, v), -1, AA)


def _blend_box(frame, x0, y0, x1, y1):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def _draw_popup(frame, hit, age):
    """着弾位置の横に点数がすっと出て消える。"""
    if age >= 0.7:
        return
    x, y = int(hit.x), int(hit.y)
    big = hit.score == 100
    a = min(1.0, age / 0.06) if age < 0.4 else max(0.0, 1 - (age - 0.4) / 0.3)
    slide = int(round(8 * (1 - min(1.0, age / 0.12)) ** 2))
    main, sc = str(hit.score), (0.95 if big else 0.8)
    (tw, th), _ = cv2.getTextSize(main, DUPLEX, sc, 2)
    w = frame.shape[1]
    left = x + 18 + tw + 12 > w
    tx = (x - 18 - tw + slide) if left else (x + 18 - slide)
    ty = y + th // 2 - (6 if big else 0)
    box = _blend_box(frame, tx - 8, ty - th - 10, tx + tw + 10, ty + 10)
    if box is None:
        return
    bx0, by0, bx1, by1 = box
    roi = frame[by0:by1, bx0:bx1]
    tmp = roi.copy()
    o = (tx - bx0, ty - by0)
    cv2.putText(tmp, main, o, DUPLEX, sc, INK, 6, AA)
    cv2.putText(tmp, main, o, DUPLEX, sc, WHITE, 2, AA)
    cv2.addWeighted(tmp, a, roi, 1 - a, 0, dst=roi)


def _draw_hit_fx(frame, st, hit, age):
    # 得点帯が一瞬だけ明るくなる(白へ寄せるだけなので色相は変わらない)
    if age < 0.28 and hit.score in st["zones"]:
        a = 0.32 * (1 - age / 0.28) ** 2
        x0, y0, x1, y1 = st["zone_box"]
        roi = frame[y0:y1, x0:x1]
        lit = cv2.addWeighted(roi, 1 - a, roi, 0, 255 * a)
        cv2.copyTo(lit, st["zones"][hit.score], roi)

    # 着弾点の細い白リング(キレ重視、0.22 秒)
    if age < 0.22:
        t = age / 0.22
        r = int(6 + 14 * (1 - (1 - t) ** 3))
        box = _blend_box(frame, hit.x - 24, hit.y - 24, hit.x + 25, hit.y + 25)
        if box:
            bx0, by0, bx1, by1 = box
            roi = frame[by0:by1, bx0:bx1]
            tmp = roi.copy()
            cv2.circle(tmp, (int(hit.x) - bx0, int(hit.y) - by0), r, WHITE, 1, AA)
            a = (1 - t) ** 1.5
            cv2.addWeighted(tmp, a, roi, 1 - a, 0, dst=roi)


# ---------------------------------------------------------------- bull (100点の祝祭演出)
# 判定円の外では BULL_END 秒続く。円内は _draw_bull_inner だけで、FX_END より前に消える。
BULL_END = 2.4
GOLD = (60, 190, 240)          # 円の外でしか使わない(橙の色相なので円内は禁止)
GOLD_DIM = (40, 120, 160)
CONFETTI = (GOLD, BOARD_RED, (90, 170, 40), (240, 240, 240))


def _bull_streak(hits, idx):
    """hits[idx] で終わるブルの連続数。"""
    n = 0
    while idx >= 0 and hits[idx].score == 100:
        n += 1
        idx -= 1
    return n


def _bull_fade(age):
    return 1.0 if age < BULL_END - 0.5 else max(0.0, (BULL_END - age) / 0.5)


def _draw_rays(frame, center, age):
    """ボードの外周から外へ伸びる光の筋。ゆっくり回りながら消える。"""
    a = min(1.0, age / 0.12) * max(0.0, 1 - age / 1.6) ** 1.5
    if a <= 0:
        return
    cx, cy = center
    lay = np.zeros_like(frame)
    rot = 0.18 * age
    n = 20
    for i in range(n):
        th = rot + (i + 0.5) * 2 * math.pi / n     # ボードの数字の間から出る
        half = math.radians(0.75 if i % 2 == 0 else 0.4)
        base = GOLD if i % 2 == 0 else (215, 235, 245)
        steps = np.linspace(QUIET_R + 2, 700 if i % 2 == 0 else 520, 13)
        for j in range(len(steps) - 1):
            r0, r1 = steps[j], steps[j + 1]
            k = (1 - j / (len(steps) - 1)) ** 1.6
            col = _scale(base, 0.75 * a * k)
            pts = [(cx + math.cos(th - half) * r0, cy + math.sin(th - half) * r0),
                   (cx + math.cos(th - half) * r1, cy + math.sin(th - half) * r1),
                   (cx + math.cos(th + half) * r1, cy + math.sin(th + half) * r1),
                   (cx + math.cos(th + half) * r0, cy + math.sin(th + half) * r0)]
            cv2.fillConvexPoly(lay, np.round(np.array(pts)).astype(np.int32), col, AA)
    cv2.add(frame, lay, dst=frame)


def _draw_confetti(frame, center, radii, hit, age):
    """ボードの縁から弾けて、ひらひら落ちる紙吹雪。ヒットごとに固定シード。"""
    a = _bull_fade(age)
    if a <= 0:
        return
    rng = np.random.default_rng(int(hit.t * 1000) & 0xFFFFFFFF)
    n = 110
    edge = _board_rings(radii)["edge"]
    ang = rng.uniform(0, 2 * np.pi, n)
    r0 = edge + rng.uniform(0, 30, n)
    spd = rng.uniform(140, 460, n)
    drag, g, fall = 2.2, 260.0, rng.uniform(60, 130, n)
    f = (1 - math.exp(-drag * age)) / drag
    sway = 14 * np.sin(rng.uniform(4, 8, n) * age + rng.uniform(0, 6.3, n))
    xs = center[0] + np.cos(ang) * (r0 + spd * f) + sway
    ys = center[1] + np.sin(ang) * (r0 + spd * f) + np.minimum(0.5 * g * age * age, fall * age)
    spin = rng.uniform(5, 12, n) * age + rng.uniform(0, 6.3, n)
    flip = np.cos(rng.uniform(6, 14, n) * age + rng.uniform(0, 6.3, n))
    size = rng.uniform(4, 7, n)
    kind = rng.integers(0, len(CONFETTI), n)
    for i in range(n):
        L, W = size[i], size[i] * 0.45 * abs(flip[i]) + 0.6
        c, s = math.cos(spin[i]), math.sin(spin[i])
        pts = np.array([[xs[i] + c * L - s * W, ys[i] + s * L + c * W],
                        [xs[i] - c * L - s * W, ys[i] - s * L + c * W],
                        [xs[i] - c * L + s * W, ys[i] - s * L - c * W],
                        [xs[i] + c * L + s * W, ys[i] + s * L - c * W]])
        shade = 0.65 + 0.35 * abs(flip[i])
        cv2.fillConvexPoly(frame, np.round(pts).astype(np.int32), _scale(CONFETTI[kind[i]], a * shade), AA)


def _draw_telop(frame, w, h, center, radii, age, streak):
    """ボードの右に出る中継テロップ風の「BULL!」。"""
    a = _bull_fade(age)
    if a <= 0:
        return
    edge = _board_rings(radii)["edge"]
    x0 = center[0] + max(edge, QUIET_R) + 18
    x1 = w - 36
    if x1 - x0 < 160:
        return
    cy = center[1]
    y0, y1 = cy - 52, cy + 52
    wipe = 1 - (1 - min(1.0, age / 0.16)) ** 3
    bx0 = int(x1 - (x1 - x0) * wipe)
    roi = frame[y0:y1, bx0:x1]
    tmp = roi.copy()
    tmp[:] = (18, 18, 20)
    cv2.addWeighted(tmp, 0.88 * a, roi, 1 - 0.88 * a, 0, dst=roi)
    for yy in (y0, y1 - 1):
        cv2.line(frame, (bx0, yy), (x1, yy), _scale(GOLD, a), 2)
    cv2.rectangle(frame, (bx0, y0), (bx0 + 5, y1 - 1), _scale(BOARD_RED, a), -1)
    if age < 0.06:
        return
    t = age - 0.06
    sc = 1.9 * (1 + 0.45 * math.exp(-t * 14) * math.cos(t * 26))   # スケールインして少し揺れる
    dx = 5 * math.exp(-t * 5) * math.sin(t * 30)
    mx = (x0 + x1) / 2 + 3 + dx
    _text_center(frame, "BULL!", mx, cy - 8, DUPLEX, sc, _scale(INK, a), 9)
    _text_center(frame, "BULL!", mx, cy - 8, DUPLEX, sc, _scale(GOLD, a), 3)
    sub = "100" if streak < 2 else f"{streak} IN A ROW"
    _text_center(frame, sub, mx, cy + 34, SIMPLEX, 0.5, _scale((235, 235, 235), a), 1)


def _draw_bull_inner(frame, st, center, radii, age):
    """円内のブル演出: 赤/緑の明滅 + 中心から広がる白いリング。0.7 秒で終わる。"""
    if age < 0.6:
        a = 0.6 * (0.5 + 0.5 * math.cos(2 * math.pi * (age - 0.1) / 0.2)) * (1 - age / 0.6)
        x0, y0, x1, y1 = st["zone_box"]
        roi = frame[y0:y1, x0:x1]
        lit = cv2.addWeighted(roi, 1 - a, roi, 0, 255 * a)
        cv2.copyTo(lit, st["zones"][100], roi)
    if age < 0.7:
        x0, y0, x1, y1 = st["zone_box"]
        roi = frame[y0:y1, x0:x1]
        c = (center[0] - x0, center[1] - y0)
        lay = np.zeros_like(roi)
        for k, delay in enumerate((0.0, 0.1, 0.2)):
            t = (age - delay) / 0.5
            if 0 < t < 1:
                r = int(radii[0] + (radii[2] + 10 - radii[0]) * (1 - (1 - t) ** 2))
                v = int(255 * (1 - t) ** 1.4 * (1.0 if k == 0 else 0.7))
                cv2.circle(lay, c, r, (v, v, v), 2, AA)
        cv2.max(roi, lay, dst=roi)   # 白いリングを「明るい方」で重ねる(色相を持ち込まない)


def _scale(col, a):
    a = max(0.0, min(1.0, a))
    return (int(col[0] * a), int(col[1] * a), int(col[2] * a))


def _last_bull(hits, now):
    """円の外の祝祭演出が続いているブル (hit, age, streak)。"""
    for idx in range(len(hits) - 1, max(-1, len(hits) - 7), -1):
        age = now - hits[idx].t
        if hits[idx].score == 100 and 0 <= age < BULL_END:
            return hits[idx], age, _bull_streak(hits, idx)
    return None


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
        _text_center(img, "CALIBRATING", w // 2, h // 2, SIMPLEX, 0.5, (0, 0, 0), 1)
        return img

    center = (int(center[0]), int(center[1]))
    radii = tuple(int(r) for r in radii)
    st = _get_static(w, h, center, radii)
    base = st["bg"]
    frame = base.copy()

    # 判定円の外で変わるもの(ブルの祝祭・スコアバー)。はみ出しても円内は静的画像に戻す
    bull = _last_bull(hits, now)
    box = _bar_rect(w, h, center, radii)
    if bull:
        hit, age, streak = bull
        _draw_rays(frame, center, age)
        _draw_confetti(frame, center, radii, hit, age)
        _draw_telop(frame, w, h, center, radii, age, streak)
        frame[box[1]:box[3] + 1, box[0]:box[2] + 1] = base[box[1]:box[3] + 1, box[0]:box[2] + 1]
    _draw_bar(frame, w, h, center, radii, total_score, hits, cooldown_progress, now)
    if bull:
        box = (center[0] - QUIET_R, center[1] - QUIET_R, center[0] + QUIET_R + 1, center[1] + QUIET_R + 1)
    sl = (slice(max(0, box[1]), min(h, box[3] + 1)), slice(max(0, box[0]), min(w, box[2] + 1)))
    cv2.copyTo(base[sl], st["quiet"][sl], frame[sl])

    # 命中演出(FX_END 秒で完全に消える) → 着弾跡 → 点数
    active = [(hit, now - hit.t) for hit in hits[-6:] if 0 <= now - hit.t < FX_END]
    for hit, age in active:
        _draw_hit_fx(frame, st, hit, age)
        if hit.score == 100:
            _draw_bull_inner(frame, st, center, radii, age)
    _draw_hit_marks(frame, hits)
    for hit, age in active:
        _draw_popup(frame, hit, age)
    return frame
