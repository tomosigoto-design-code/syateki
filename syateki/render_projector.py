"""投影画面の描画(3色リングの的 + 中継風スコアバー)。
mode="fever" ではブルがパチンコ風に派手になり、2倍タイム(bonus_start〜bonus_until)は龍と炎のリングが出る。
右下の常設ランプは、ブルで「ペカッ」と点灯する(normal は控えめ、fever は背景ごと)。

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
    mult: int = 1  # 2倍タイム中の命中は 2。表示する点数は score * mult(色や帯は score で決める)


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
    _draw_lamp_off(bg, w, h)
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
        _prewarm_glyphs()
    return v


# ---------------------------------------------------------------- score bar (判定円の外)
def _draw_reel(frame, txt, right, base_y, age, col):
    """スロットのリール風: 各桁が高速で切り替わり、左から順に止まる。"""
    sc, th = 1.55, 2
    cell = max(cv2.getTextSize(d, DUPLEX, sc, th)[0][0] for d in "0123456789") + 2
    ch = cv2.getTextSize("0", DUPLEX, sc, th)[0][1]
    x0 = right - cell * len(txt)
    y_top, y_bot = base_y - ch - 6, base_y + 6
    win = frame[y_top:y_bot, x0:right]
    tmp = win.copy()
    for i, d in enumerate(txt):
        stop = 0.55 + 0.28 * i
        x = i * cell
        if age >= stop:
            settle = max(0.0, 1 - (age - stop) / 0.12)
            dy = int(-3 * settle * math.cos((age - stop) * 40))
            cv2.putText(tmp, d, (x, ch + 6 + dy), DUPLEX, sc, col, th, AA)
        else:   # 回転中: 1桁ずつ素早く切り替える(隣の数字は出さないので重ならない)
            pos = age * 22 + i * 3.7
            k = int(pos) % 10
            cv2.putText(tmp, str(k), (x, ch + 6), DUPLEX, sc, (170, 170, 170), th, AA)
    win[:] = tmp      # 窓の外へはみ出した分は捨てる(パネル内でクリップ)


def _draw_bar(frame, w, h, center, radii, total_score, hits, cooldown_progress, now, fever=None):
    bx0, by0, bx1, by1 = _bar_rect(w, h, center, radii)
    shown = total_score
    col = WHITE
    if fever is not None:
        age = fever
        stop_all = 0.55 + 0.28 * (len(str(total_score)) - 1) + 0.15
        col = GOLD if age < stop_all or int(age * 10) % 2 else (255, 255, 255)
        if age > FEVER_END - 0.6:
            col = GOLD
        _draw_reel(frame, str(max(0, total_score)), bx1 - 16, by0 + 66, age, col)
    elif hits:
        age = now - hits[-1].t
        dur = COUNT_UP * 2 if hits[-1].score == 100 else COUNT_UP
        if 0 <= age < dur:
            p = 1 - (1 - age / dur) ** 2
            shown = int(round(total_score - hits[-1].score * hits[-1].mult * (1 - p)))
        if hits[-1].score == 100 and 0 <= age < BULL_END:
            k = 1.0 if age < 1.2 else max(0.0, 1 - (age - 1.2) / (BULL_END - 1.2))
            col = tuple(int(g * k + 255 * (1 - k)) for g in GOLD)
    if fever is None:
        txt = str(max(0, shown))
        (tw, _), _ = cv2.getTextSize(txt, DUPLEX, 1.55, 2)
        cv2.putText(frame, txt, (bx1 - 16 - tw, by0 + 66), DUPLEX, 1.55, col, 2, AA)

    mid = (bx0 + bx1) // 2
    cv2.putText(frame, str(len(hits)), (bx0 + 16, by0 + 118), DUPLEX, 0.7, WHITE, 1, AA)
    if hits:
        cv2.putText(frame, str(hits[-1].score * hits[-1].mult), (mid + 8, by0 + 118), DUPLEX, 0.7, WHITE, 1, AA)

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
    if hit.mult > 1:
        main = f"+{hit.score * hit.mult} x{hit.mult}"
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


def _draw_confetti(frame, center, radii, hit, age, n=110, palette=CONFETTI, a=None, seed=0):
    """ボードの縁から弾けて、ひらひら落ちる紙吹雪。ヒットごとに固定シード。"""
    a = _bull_fade(age) if a is None else a
    if a <= 0:
        return
    rng = np.random.default_rng((int(hit.t * 1000) + seed) & 0xFFFFFFFF)
    edge = _board_rings(radii)["edge"]
    ang = rng.uniform(0, 2 * np.pi, n)
    r0 = edge + rng.uniform(0, 30, n)
    spd = rng.uniform(140, 460, n)
    drag, g, fall = 2.2, 260.0, rng.uniform(60, 130, n)
    f = (1 - math.exp(-drag * age)) / drag
    sway = 14 * np.sin(rng.uniform(4, 8, n) * age + rng.uniform(0, 6.3, n))
    xs = center[0] + np.cos(ang) * (r0 + spd * f) + sway
    ys = center[1] + np.sin(ang) * (r0 + spd * f) + np.minimum(0.5 * g * age * age, fall * age)
    _draw_flakes(frame, rng, xs, ys, age, palette, a, center)


def _draw_flakes(frame, rng, xs, ys, age, palette, a, center=None):
    """紙吹雪の小片(回転しながら裏返る長方形)を描く。center を渡すと判定円の中の分は描かない(どうせ戻すので)。"""
    n = len(xs)
    spin = rng.uniform(5, 12, n) * age + rng.uniform(0, 6.3, n)
    flip = np.abs(np.cos(rng.uniform(6, 14, n) * age + rng.uniform(0, 6.3, n)))
    size = rng.uniform(4, 7, n)
    kind = rng.integers(0, len(palette), n)
    h, w = frame.shape[:2]
    keep = (xs > -10) & (xs < w + 10) & (ys > -10) & (ys < h + 10)
    if center is not None:
        keep &= np.hypot(xs - center[0], ys - center[1]) > QUIET_R - 12
    idx = np.nonzero(keep)[0]
    if len(idx) == 0:
        return
    L, Wd = size[idx], size[idx] * 0.45 * flip[idx] + 0.6
    c, s_ = np.cos(spin[idx]), np.sin(spin[idx])
    x, y = xs[idx], ys[idx]
    pts = np.stack([np.stack([x + c * L - s_ * Wd, y + s_ * L + c * Wd], 1),
                    np.stack([x - c * L - s_ * Wd, y - s_ * L + c * Wd], 1),
                    np.stack([x - c * L + s_ * Wd, y - s_ * L - c * Wd], 1),
                    np.stack([x + c * L + s_ * Wd, y + s_ * L - c * Wd], 1)], 1).round().astype(np.int32)
    pal = np.array(palette, np.float32)[kind[idx]] * (a * (0.65 + 0.35 * flip[idx]))[:, None]
    cols = pal.astype(np.int32).tolist()
    for i in range(len(idx)):
        cv2.fillConvexPoly(frame, pts[i], cols[i], AA)


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


def _last_bull(hits, now, end=BULL_END):
    """円の外の祝祭演出が続いているブル (hit, age, streak)。"""
    for idx in range(len(hits) - 1, max(-1, len(hits) - 7), -1):
        age = now - hits[idx].t
        if hits[idx].score == 100 and 0 <= age < end:
            return hits[idx], age, _bull_streak(hits, idx)
    return None


# ---------------------------------------------------------------- fever (パチンコ風のブル演出)
# mode="fever" のときだけ。円の外で FEVER_END 秒。円内は通常のブルと同じ(_draw_bull_inner, FX_END で消える)。
# 露出対策: 白一色の面は出さない。明るさではなく色・動き・量で派手にする。
FEVER_END = 3.6
_HUE_LUT = cv2.cvtColor(np.array([[[hh, 255, 255] for hh in range(180)]], np.uint8), cv2.COLOR_HSV2BGR)[0]
RAINBOW = tuple(tuple(int(v) for v in _HUE_LUT[hh]) for hh in range(0, 180, 20)) + (GOLD, (240, 240, 240))
_glyph_cache: dict = {}


_HUE_T = [tuple(int(v) for v in c) for c in _HUE_LUT]


def _hue(hh, v=1.0):
    c = _HUE_T[int(hh) % 180]
    if v >= 1.0:
        return c
    v = max(0.0, v)
    return (int(c[0] * v), int(c[1] * v), int(c[2] * v))


def _fever_fade(age):
    return 1.0 if age < FEVER_END - 0.5 else max(0.0, (FEVER_END - age) / 0.5)


def _crop(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _ff_mask(text, px):
    """OpenCV 5 の FontFace で text を描いた 1ch マスク。描けなければ None。"""
    try:
        ff = cv2.FontFace("sans")
        img = np.zeros((px * 2, px * (len(text) + 2)), np.uint8)
        cv2.putText(img, text, (px // 4, int(px * 1.4)), (255,), ff, px, 800)
        q = np.zeros_like(img)
        cv2.putText(q, "?" * len(text), (px // 4, int(px * 1.4)), (255,), ff, px, 800)
        if text != "?" * len(text) and np.array_equal(img, q):
            return None   # 字形が無くて ? に置き換わった
        return _crop(img)
    except Exception:
        return None


def _hershey_mask(text, scale, thick):
    (tw, th), bl = cv2.getTextSize(text, DUPLEX, scale, thick)
    img = np.zeros((th + bl + 2 * thick + 4, tw + 2 * thick + 4), np.uint8)
    cv2.putText(img, text, (thick + 2, th + thick + 2), DUPLEX, scale, 255, thick, AA)
    return _crop(img)


def _jackpot_glyphs(px):
    """縦書きの「大当たり!!」の字形マスク(太らせ済み, 縁取り)のリスト。日本語不可なら JACKPOT!!。"""
    key = ("jp", px)
    if key not in _glyph_cache:
        jp = ["大", "当", "た", "り", "!!"]
        masks = [_ff_mask(c, px) for c in jp]
        if any(m is None for m in masks):
            masks = [_hershey_mask(c, px / 34, max(2, px // 14)) for c in "JACKPOT"] + \
                    [_hershey_mask("!!", px / 34, max(2, px // 14))]
            masks = [cv2.resize(m, None, fx=0.62, fy=0.62, interpolation=cv2.INTER_AREA) for m in masks]
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        out = []
        for m in masks:
            m = cv2.dilate(np.pad(m, 12), k)                      # 太らせる
            edge = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
            out.append((m, edge))
        _glyph_cache[key] = out
    return _glyph_cache[key]


def _small_mask(text, px):
    key = ("s", text, px)
    if key not in _glyph_cache:
        m = _ff_mask(text, px)
        if m is None:
            m = _hershey_mask(text.replace("×", "x"), px / 30, max(2, px // 16))
        m = np.pad(m, 6)
        _glyph_cache[key] = (m, cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))))
    return _glyph_cache[key]


def _paste_mask(frame, mask, edge, cx, cy, scale, fill, edge_col=(10, 10, 14)):
    """mask を (cx, cy) 中心に scale 倍で置く。fill は BGR か (gh, gw, 3) の色画像。"""
    if scale != 1.0:
        sz = (max(1, int(mask.shape[1] * scale)), max(1, int(mask.shape[0] * scale)))
        mask = cv2.resize(mask, sz, interpolation=cv2.INTER_LINEAR)
        edge = cv2.resize(edge, sz, interpolation=cv2.INTER_LINEAR)
    gh, gw = mask.shape
    x0, y0 = int(cx - gw / 2), int(cy - gh / 2)
    H, W = frame.shape[:2]
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + gw), min(H, y0 + gh)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sub = (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))
    roi = frame[fy0:fy1, fx0:fx1]
    rh, rw = roi.shape[:2]
    e = cv2.threshold(edge[sub], 100, 255, cv2.THRESH_BINARY)[1]
    cv2.subtract(roi, (255, 255, 255, 0), dst=roi, mask=e)            # 縁は黒
    if any(edge_col):
        cv2.add(roi, edge_col, dst=roi, mask=e)
    m = cv2.threshold(mask[sub], 110, 255, cv2.THRESH_BINARY)[1]
    if isinstance(fill, np.ndarray):                                   # 縦グラデーション (gh', 1, 3)
        col = cv2.resize(fill, (gw, gh), interpolation=cv2.INTER_NEAREST)[sub]
        cv2.copyTo(col, m, roi)
    else:
        cv2.subtract(roi, (255, 255, 255, 0), dst=roi, mask=m)
        cv2.add(roi, fill, dst=roi, mask=m)


def _prewarm_glyphs():
    """fever の大文字の字形を先に作っておく(初回の命中フレームで描画時間が跳ねないように)。"""
    _jackpot_glyphs(118)
    _small_mask("+100", 80)
    for n in range(2, 7):
        _small_mask(f"×{n}", 96)
    _small_mask("×2", 130)


def _draw_sunburst(frame, center, age, wedges=True):
    """回転する放射バースト(色の帯)+ 虹色に流れる細い光の筋。円内は後で静的画像に戻る。"""
    a = min(1.0, age / 0.1) * _fever_fade(age)
    if a <= 0:
        return
    cx, cy = center
    lay = frame          # 背景はほぼ一様なチャコールなので、BG + 色 を直接描く
    R = 1500
    n = 16
    rot = 0.9 * age
    for i in range(n if wedges else 0):
        if i % 2:
            continue
        t0 = rot + i * 2 * math.pi / n
        t1 = t0 + 2 * math.pi / n
        pts = np.array([[cx, cy], [cx + math.cos(t0) * R, cy + math.sin(t0) * R],
                        [cx + math.cos(t1) * R, cy + math.sin(t1) * R]], np.int32)
        c = _hue(i * 11 + age * 70, 0.17 * a)
        cv2.fillConvexPoly(lay, pts, tuple(min(255, BG[k] + c[k]) for k in range(3)), AA)
    # 虹色の光の筋(逆回転、色が根元から外へ流れる)
    m = 30
    for i in range(m):
        th = -1.4 * age + i * 2 * math.pi / m
        c, s_ = math.cos(th), math.sin(th)
        for j in range(5):
            r0, r1 = QUIET_R + 6 + j * 110, QUIET_R + 6 + (j + 1) * 110
            col = _hue(i * 6 + j * 30 - age * 260, 0.9 * a * (1 - j / 6))
            cv2.line(lay, (int(cx + c * r0), int(cy + s_ * r0)), (int(cx + c * r1), int(cy + s_ * r1)),
                     col, 2 if i % 3 else 3, AA)


def _draw_stars(frame, center, hit, age, n=60):
    """星のきらめき(4本のとげの星が瞬く)。円の外の固定位置。"""
    a = _fever_fade(age)
    rng = np.random.default_rng((int(hit.t * 1000) + 7) & 0xFFFFFFFF)
    h, w = frame.shape[:2]
    pts = rng.uniform((0, 0), (w, h), (160, 2))
    d = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
    pts = pts[d > QUIET_R + 20][:n]
    ph = rng.uniform(0, 6.3, len(pts))
    om = rng.uniform(7, 14, len(pts))
    sz = rng.uniform(6, 15, len(pts))
    kind = rng.integers(0, 3, len(pts))
    for i, (x, y) in enumerate(pts):
        k = math.sin(ph[i] + om[i] * age)
        if k <= 0.15:
            continue
        r, q = sz[i] * k, sz[i] * k * 0.22
        star = np.array([[x, y - r], [x + q, y - q], [x + r, y], [x + q, y + q],
                         [x, y + r], [x - q, y + q], [x - r, y], [x - q, y - q]], np.int32)
        col = ((245, 245, 245), GOLD, _hue(age * 120 + i * 13))[kind[i]]
        cv2.fillPoly(frame, [star], _scale(col, a), AA)


def _draw_rain(frame, hit, age):
    """上から降る紙吹雪。"""
    a = _fever_fade(age)
    rng = np.random.default_rng((int(hit.t * 1000) + 3) & 0xFFFFFFFF)
    h, w = frame.shape[:2]
    n = 170
    x0 = rng.uniform(0, w, n)
    y0 = rng.uniform(-h * 1.2, -10, n)
    vy = rng.uniform(150, 280, n)
    xs = x0 + 22 * np.sin(rng.uniform(2, 5, n) * age + rng.uniform(0, 6.3, n))
    ys = y0 + vy * age
    _draw_flakes(frame, rng, xs, ys, age, RAINBOW, a)


def _draw_marquee(frame, age, step=30):
    """画面外周の電飾(色が流れて、チカチカ点滅する)。"""
    a = _fever_fade(age)
    h, w = frame.shape[:2]
    m = 13
    pts = [(x, m) for x in range(m, w - m, step)] + [(w - m, y) for y in range(m, h - m, step)] + \
          [(x, h - m) for x in range(w - m, m, -step)] + [(m, y) for y in range(h - m, m, -step)]
    tick = int(age * 16)
    for i, p in enumerate(pts):
        on = (i + tick) % 3 != 0
        if tick % 6 >= 4:              # ときどき全灯と交互点滅
            on = (i + tick) % 2 == 0
        if on:
            cv2.circle(frame, p, 5, _hue(i * 9 + age * 300, a), -1, AA)
        else:
            cv2.circle(frame, p, 4, _scale((60, 60, 70), a), -1, AA)


def _draw_jackpot(frame, w, h, center, age):
    """巨大な縦書き「大当たり!!」。1文字ずつドーンと出て、揺れて、虹色に流れる。"""
    a = _fever_fade(age)
    if a <= 0:
        return
    x_l = center[0] + QUIET_R + 10
    cx = (x_l + w - 30) / 2
    glyphs = _jackpot_glyphs(118)
    n = len(glyphs)
    total = sum(m.shape[0] for m, _ in glyphs) - 24 * n + 8 * (n - 1)
    top, bot = 40, h - 200            # 右下のランプを避ける
    sc = min(1.0, (bot - top) / total, (w - 30 - x_l) / max(m.shape[1] for m, _ in glyphs))
    y = top + (bot - top - total * sc) / 2
    for i, (m, e) in enumerate(glyphs):
        gh = (m.shape[0] - 24) * sc
        t = age - 0.05 - 0.08 * i
        y_c = y + gh / 2
        y += gh + 8 * sc
        if t < 0:
            continue
        pop = 1 + 0.9 * math.exp(-t * 12) * abs(math.cos(t * 18))          # ドーン
        shake = 6 * math.sin(age * 41 + i * 2) * (0.35 + 0.65 * math.exp(-t * 3))
        jitter = 3 * math.sin(age * 57 + i)
        hue0 = age * 200 - i * 25
        rows = np.arange(m.shape[0]) * 1.2
        grad = _HUE_LUT[((hue0 + rows) % 180).astype(np.int32)][:, None, :]
        fill = grad if a >= 1 else (grad.astype(np.float32) * a).astype(np.uint8)
        _paste_mask(frame, m, e, cx + shake, y_c + jitter, sc * pop, fill, _scale((10, 10, 14), a))


def _draw_fever_telops(frame, w, h, center, age, streak):
    """CONGRATULATIONS(左上)と ×2 / +100(左下)。"""
    a = _fever_fade(age)
    if a <= 0:
        return
    if age > 0.45:
        t = age - 0.45
        txt = "CONGRATULATIONS"
        (tw, th), _ = cv2.getTextSize(txt, DUPLEX, 0.9, 2)
        x0, y0 = 70, 96
        wipe = min(1.0, t / 0.2)
        bw = int((tw + 36) * wipe)
        cv2.rectangle(frame, (x0, y0), (x0 + bw, y0 + 48), _scale((20, 14, 26), a), -1)
        for yy in (y0, y0 + 48):
            for k in range(0, bw, 12):
                cv2.line(frame, (x0 + k, yy), (x0 + min(bw, k + 12), yy), _hue(k + age * 400, a), 2)
        if wipe >= 1:
            x = x0 + 18
            for j, ch in enumerate(txt):
                bob = int(4 * math.sin(age * 12 - j * 0.6))
                cv2.putText(frame, ch, (x, y0 + 34 + bob), DUPLEX, 0.9, _hue(j * 12 + age * 250, a), 2, AA)
                x += cv2.getTextSize(ch, DUPLEX, 0.9, 2)[0][0]
    if age > 0.8:
        t = age - 0.8
        txt = f"×{streak}" if streak >= 2 else "+100"
        m, e = _small_mask(txt, 96 if streak >= 2 else 80)
        pop = 1 + 0.8 * math.exp(-t * 10) * abs(math.cos(t * 16))
        flash = GOLD if int(age * 8) % 2 else (255, 255, 255)
        _paste_mask(frame, m, e, 200, h - 150 + 4 * math.sin(age * 30), pop, _scale(flash, a),
                    _scale((10, 10, 14), a))
        if streak >= 2:
            cv2.putText(frame, "BULL COMBO", (128, h - 88), DUPLEX, 0.7, _scale(INK, a), 6, AA)
            cv2.putText(frame, "BULL COMBO", (128, h - 88), DUPLEX, 0.7, _hue(age * 300, a), 2, AA)


def _draw_fever_badge(frame, w, h, center, radii):
    """fever 中の小さなバッジ(静止)。スコアバー凡例の下。"""
    bx0, _, _, by1 = _bar_rect(w, h, center, radii)
    y = by1 + 66
    cv2.rectangle(frame, (bx0, y), (bx0 + 112, y + 24), (40, 20, 150), -1)
    cv2.putText(frame, "FEVER MODE", (bx0 + 9, y + 17), SIMPLEX, 0.45, (230, 240, 255), 1, AA)


def _draw_fever_outer(frame, w, h, center, radii, hit, age, streak, dragon=False, fire=False):
    """dragon=True(2倍タイム中)は龍と炎が主役なので、ブルの外側演出は省く(描画時間・輝度対策)。"""
    if not dragon:
        _draw_sunburst(frame, center, age, wedges=False)   # 背景の色面はランプのペカりが担う(平均輝度対策)
    if not dragon:
        _draw_stars(frame, center, hit, age)
        _draw_confetti(frame, center, radii, hit, age, n=140 if fire else 200, palette=RAINBOW, a=_fever_fade(age),
                       seed=11)
    if not dragon:
        _draw_rain(frame, hit, age)
    if not dragon:
        _draw_marquee(frame, age)
    if not dragon:
        _draw_fever_telops(frame, w, h, center, age, streak)
        _draw_jackpot(frame, w, h, center, age)


# ---------------------------------------------------------------- bonus (fever の2倍タイム: 龍 + 炎のリング)
# bonus_start <= now < bonus_until の間だけ。龍・炎・テロップはすべて半径300の外(描いた後に円内を静的画像へ戻す)。
DRAGON_T = 3.0           # 発動演出(入場・ブレス・印・稲妻・テロップ)の長さ。龍そのものは bonus_until まで舞う
DRAGON_R0 = 350          # 龍が回る軌道の半径
DRAGON_N = 44            # 胴の節の数
DRAGON_EXIT = 1.0        # 終了何秒前から画面外へ去るか
FIRE_FADE = 1.0          # 終了何秒前から炎が弱まるか
OMEGA, OMEGA_X = 2.1, 4.8    # 周回 / 退場の角速度 (rad/s)
D_RED = (34, 36, 196)
D_RED_DK = (20, 20, 110)
D_RED_HI = (70, 80, 240)
D_GOLD = (50, 185, 245)
D_GOLD_DK = (20, 110, 170)
D_CREAM = (170, 225, 245)
PAL_RED = {"main": D_RED, "dk": D_RED_DK, "hi": D_RED_HI, "belly": D_GOLD, "scale": D_GOLD_DK,
           "fin": D_GOLD, "fin2": D_RED, "snout": D_GOLD, "glow": (30, 40, 170)}
PAL_GOLD = {"main": (30, 140, 205), "dk": (15, 70, 125), "hi": (90, 195, 240), "belly": (140, 200, 225),
            "scale": (20, 110, 170), "fin": D_RED, "fin2": (30, 120, 210), "snout": D_CREAM, "glow": (30, 140, 200)}
PHI_START = math.pi * 0.62 - 0.9


def _dragon_phase(t, total):
    """頭の角度 Φ(t) と退場開始の角度 Φ(te)。周回は OMEGA、退場は OMEGA_X で加速する。t は配列でもよい。"""
    te = total - DRAGON_EXIT
    phi = PHI_START + OMEGA * t + (OMEGA_X - OMEGA) * np.maximum(0.0, t - te)
    return phi, PHI_START + OMEGA * te


def _dragon_path(center, t, total, sgn):
    """龍の頭が時刻 t(配列)にいる位置。外から渦を巻いて入り、外周を回り、最後は渦を巻いて画面外へ去る。
    sgn=±1 でうねりを逆位相にして、2匹が絡み合うように交差させる。"""
    t = np.asarray(t, np.float64)
    phi, _ = _dragon_phase(t, total)
    te = total - DRAGON_EXIT
    r = DRAGON_R0 + 650 * np.clip((0.45 - t) / 0.45, 0, None) ** 1.2 + 900 * np.maximum(0.0, t - te) ** 1.4
    r = r + sgn * 17 * np.sin(t * 9)
    return np.stack([center[0] + np.cos(phi) * r, center[1] + np.sin(phi) * r], 1)


def _dragon_geom(center, t, total, sgn, size, n=DRAGON_N, seg=18.0):
    """胴の節。頭の過去の軌跡を一定の間隔(弧長 seg)でたどる(列車のように、どの速さでも胴の長さは一定)。"""
    hist = _dragon_path(center, t - np.arange(0, 220) * 0.008, total, sgn)
    d = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(hist, axis=0).T))])
    s_ = np.arange(n + 1) * seg * size
    P = np.stack([np.interp(s_, d, hist[:, 0]), np.interp(s_, d, hist[:, 1])], 1)
    tang = P[:-1] - P[1:]
    tang /= np.maximum(1e-6, np.hypot(tang[:, 0], tang[:, 1]))[:, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    flip = np.sign((nrm * (P[:-1] - np.array(center))).sum(1))[:, None]   # 背中は外向き
    nrm = nrm * np.where(flip == 0, 1, flip)
    j = np.arange(n)
    width = size * 26 * (0.7 + 0.3 * np.sin(np.clip(j / 8, 0, 1) * math.pi / 2)) * (1 - 0.82 * (j / n) ** 1.3)
    return P, tang, nrm, width


def _i32(a):
    return np.round(a).astype(np.int32)


def _draw_dragon_body(frame, P, tang, nrm, width, t, pal, hs, legs=True, detail=True):
    """胴は太さの変わる帯を多角形で塗る(縁取り → 胴 → 背の光沢 → 腹 → 鱗)、背びれ・脚・尾の房。"""
    N = DRAGON_N
    Q = P[:N]
    Wn = nrm * width[:, None]
    ip = lambda v: (int(round(v[0])), int(round(v[1])))
    ks = np.arange(2, N - 1, 2)
    fl = width[ks] * 0.8 + 5 + 3 * np.sin(t * 12 + ks)
    T, Nn = tang[ks], nrm[ks]
    base = Q[ks] + Nn * (width[ks] * 0.5)[:, None]
    tip = Q[ks] + Nn * (width[ks] + fl)[:, None] - T * 15
    fins = np.stack([base + T * 10, tip, base - T * 12], 1)
    inner = np.stack([fins[:, 0], (fins[:, 0] + fins[:, 1]) / 2, fins[:, 2]], 1)
    alt = (np.arange(len(ks)) % 2) == 0
    for sel in (alt, ~alt):   # 隣どうしは重なるので分ける(fillPoly は重なりを穴にする)
        cv2.fillPoly(frame, list(_i32(fins[sel])), pal["dk"], AA)
    if detail:
        cv2.fillPoly(frame, list(_i32(inner[alt])), pal["fin"], AA)
        cv2.fillPoly(frame, list(_i32(inner[~alt])), pal["fin2"], AA)
    for k in ((10, 27) if legs else ()):                 # 脚(腹側に2対、爪)
        (tx, ty), (nx, ny), (x, y), w_ = tang[k], nrm[k], Q[k], width[k]
        for d in (-6, 6):
            sw = 6 * math.sin(t * 14 + k + d)
            b = (x - nx * w_ * 0.6 + tx * d, y - ny * w_ * 0.6 + ty * d)
            kn = (b[0] - nx * 19 + tx * (sw - 7), b[1] - ny * 19 + ty * (sw - 7))
            ft = (kn[0] - nx * 4 + tx * 11, kn[1] - ny * 4 + ty * 11)
            cv2.line(frame, ip(b), ip(kn), pal["dk"], 8, AA)
            cv2.line(frame, ip(b), ip(kn), pal["main"], 5, AA)
            cv2.line(frame, ip(kn), ip(ft), pal["main"], 5, AA)
            for c_ in (-1, 0, 1):
                cl = (ft[0] - nx * (6 - c_ * 2) + tx * (5 + c_ * 3), ft[1] - ny * (6 - c_ * 2) + ty * (5 + c_ * 3))
                cv2.line(frame, ip(ft), ip(cl), D_CREAM, 2, AA)

    def band(off, half, col, outline=None):
        c_ = Q + Wn * off
        poly = _i32(np.concatenate([c_ + Wn * half, (c_ - Wn * half)[::-1]]))
        cv2.fillPoly(frame, [poly], col, AA)
        if outline is not None:
            cv2.polylines(frame, [poly], True, outline, 2, AA)

    band(0, 1.0, pal["main"], pal["dk"])
    band(0.38, 0.42, pal["hi"])
    band(-0.58, 0.36, pal["belly"])
    kk = np.arange(1, N - 1, 2 if detail else 4)
    arc = np.linspace(-0.9, 0.9, 5)
    c_ = Q[kk] + Wn[kk] * 0.15
    r_ = (width[kk] * 0.55)[:, None, None]
    T2, N2 = tang[kk][:, None, :], nrm[kk][:, None, :]
    scales = c_[:, None, :] + (-T2 * np.cos(arc)[None, :, None] + N2 * np.sin(arc)[None, :, None]) * r_
    cv2.polylines(frame, list(_i32(scales)), False, pal["scale"], 1, AA)
    bc = Q[kk] - Wn[kk] * 0.58
    cv2.polylines(frame, list(_i32(np.stack([bc + Wn[kk] * 0.33, bc - Wn[kk] * 0.33], 1))), False, pal["scale"], 1, AA)
    xt, yt = P[N - 1]
    tx, ty = tang[N - 2]
    nx, ny = nrm[N - 2]
    for s_ in (-1, 0, 1):                                # 尾の房
        sw = 6 * math.sin(t * 13 + s_)
        tipp = (xt - tx * 40 + nx * (s_ * 16 + sw), yt - ty * 40 + ny * (s_ * 16 + sw))
        cv2.fillConvexPoly(frame, _i32(np.array([[xt + nx * 6, yt + ny * 6], tipp, [xt - nx * 6, yt - ny * 6]])),
                           pal["fin"] if s_ else pal["fin2"], AA)
    _draw_dragon_head(frame, P[0] + tang[0] * 6, tang[0], nrm[0], t, pal, hs)


def _draw_dragon_head(frame, p, f, n, t, pal=PAL_RED, HS=1.6):
    x, y = p
    fx, fy = f
    nx, ny = n

    def q(a, b):                                       # 頭の座標系 (前方 a, 背側 b)
        return (int(x + (fx * a + nx * b) * HS), int(y + (fy * a + ny * b) * HS))

    def e(a, b, ax, ay, a0, a1, col, th=-1):
        cv2.ellipse(frame, q(a, b), (int(ax * HS), int(ay * HS)), ang, a0, a1, col, th, AA)

    ang = math.degrees(math.atan2(fy, fx))
    for i in range(6):                                 # たてがみ(後ろへ流れる)
        sway = 5 * math.sin(t * 14 + i)
        cv2.fillConvexPoly(frame, np.array([q(-2 - i * 6, 12 - i), q(-30 - i * 7, 26 + i * 2 + sway), q(-14 - i * 6, 2)],
                                           np.int32), pal["dk"] if i % 2 else pal["main"], AA)
        cv2.fillConvexPoly(frame, np.array([q(-6 - i * 6, 9), q(-24 - i * 7, 20 + i * 2 + sway), q(-14 - i * 6, 4)],
                                           np.int32), pal["fin"], AA)
    for off, col in ((0, D_CREAM), (8, D_GOLD)):        # 角(2本、後ろへ反る)
        horn = [q(-2 - off - k * 6, 15 + k * 4.5 - k * k * 0.55) for k in range(7)]
        for k in range(6):
            cv2.line(frame, horn[k], horn[k + 1], col, max(2, int((6 - k) * HS * 0.8)), AA)
    gape = 8 + 5 * math.sin(t * 12)                    # 下あご・舌・牙
    cv2.fillConvexPoly(frame, np.array([q(0, -6), q(40, -12 - gape), q(36, -2), q(10, 0)], np.int32), (25, 15, 100), AA)
    cv2.fillConvexPoly(frame, np.array([q(-2, -6), q(38, -14 - gape), q(35, -21 - gape), q(2, -17)], np.int32),
                       pal["dk"], AA)
    cv2.fillConvexPoly(frame, np.array([q(-1, -8), q(36, -15 - gape), q(33, -19 - gape), q(3, -15)], np.int32),
                       pal["main"], AA)
    cv2.line(frame, q(14, -5 - gape * 0.3), q(46, -8 - gape * 0.4), (60, 60, 230), max(2, int(2 * HS)), AA)
    for k in range(3):
        cv2.fillConvexPoly(frame, np.array([q(16 + k * 7, -2), q(18 + k * 7, -8), q(21 + k * 7, -2)], np.int32),
                           (235, 240, 245), AA)
    cv2.fillConvexPoly(frame, np.array([q(30, -14 - gape), q(32, -8 - gape), q(34, -15 - gape)], np.int32),
                       (235, 240, 245), AA)
    e(0, 2, 24, 17, 0, 360, pal["dk"])                 # 頭蓋・鼻づら
    e(0, 2, 22, 15, 0, 360, pal["main"])
    e(-2, 7, 14, 7, 0, 360, pal["hi"])
    e(25, 1, 20, 10, 0, 360, pal["dk"])
    e(25, 1, 19, 9, 0, 360, pal["snout"])
    e(26, 3, 15, 5, 0, 360, D_CREAM)
    cv2.circle(frame, q(41, 5), max(2, int(2 * HS)), (20, 20, 40), -1, AA)
    e(4, 10, 12, 6, 180, 360, pal["fin"], max(2, int(3 * HS)))     # 眉と目
    cv2.circle(frame, q(5, 7), int(6 * HS), (20, 20, 40), -1, AA)
    cv2.circle(frame, q(5, 7), int(5 * HS), (40, 215, 250), -1, AA)
    cv2.ellipse(frame, q(6, 7), (max(1, int(1.3 * HS)), int(4.5 * HS)), ang + 90, 0, 360, (10, 10, 10), -1, AA)
    cv2.circle(frame, q(3, 9), max(1, int(1.2 * HS)), (255, 255, 255), -1, AA)
    for side in (1, -1):                               # ひげ
        wh = [q(42 + 10 * math.sin(k * 0.35) - k * 4.5, side * (4 + k * 2.6) + 5 * math.sin(k * 0.7 - t * 11 + side))
              for k in range(13)]
        cv2.polylines(frame, [np.array(wh, np.int32)], False, D_GOLD, 2, AA)


def _mouth(P, tang, nrm, hs):
    """頭の口先の位置(ブレスの出どころ)。"""
    p = P[0] + tang[0] * 6
    return p + tang[0] * 42 * hs - nrm[0] * 12 * hs


def _breath_dir(tang, nrm):
    """ブレスの向き: 前方やや内側(炎のリングに当たる向き)。"""
    d = tang[0] * 0.85 - nrm[0] * 0.5
    return d / np.hypot(*d)


def _draw_breath(frame, glow, gs, src, f, t, k):
    """火炎ブレス(口から前方やや内側へ広がる炎の粒)。加算で重ねるので、消えかけの粒は透けて消える。"""
    n = 16
    i = np.arange(n)
    life = (t * 3.0 + i / n) % 1.0
    d = 12 + life * 170
    spread = (np.sin(i * 2.3 + t * 9) * 0.5) * life * 60
    nrm = np.array([-f[1], f[0]])
    pts = src[None, :] + f[None, :] * d[:, None] + nrm[None, :] * spread[:, None]
    rad = 4 + life * 16
    cols = [_scale((90, 235, 255) if lj < 0.25 else ((30, 150, 255) if lj < 0.6 else (30, 60, 230)),
                   k * (1 - lj) ** 0.8) for lj in life]
    if glow is not None:
        for j in range(n):
            cv2.circle(glow, (int(pts[j, 0] * gs), int(pts[j, 1] * gs)), max(1, int(rad[j] * gs * 1.6)), cols[j], -1)
    if frame is None:
        return
    H, W = frame.shape[:2]
    x0, y0 = max(0, int((pts[:, 0] - rad).min()) - 2), max(0, int((pts[:, 1] - rad).min()) - 2)
    x1, y1 = min(W, int((pts[:, 0] + rad).max()) + 3), min(H, int((pts[:, 1] + rad).max()) + 3)
    if x1 <= x0 or y1 <= y0:
        return
    lay = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
    for j in np.argsort(-life):
        cv2.circle(lay, (int(pts[j, 0]) - x0, int(pts[j, 1]) - y0), int(rad[j]), cols[j], -1, AA)
    roi = frame[y0:y1, x0:x1]
    cv2.add(roi, lay, dst=roi)


def _draw_pearl(frame, glow, gs, p, t):
    """龍が追う宝珠(火の玉)。"""
    x, y = int(p[0]), int(p[1])
    if glow is not None:
        cv2.circle(glow, (int(x * gs), int(y * gs)), max(2, int(34 * gs)), (40, 170, 240), -1)
    if frame is None:
        return
    for i in range(3):                                 # 宝珠を巻く炎
        a0 = t * 400 + i * 120
        cv2.ellipse(frame, (x, y), (22, 22), a0, 0, 80, (20, 120, 250), 4, AA)
    cv2.circle(frame, (x, y), 15, (30, 150, 245), -1, AA)
    cv2.circle(frame, (x, y), 11, (150, 235, 255), -1, AA)
    cv2.circle(frame, (x - 4, y - 4), 4, (245, 252, 255), -1, AA)


def _trail_points(center, t, total, sgn, n, seed):
    """頭が通った跡の光の粒の位置(過去の頭の位置 + 時刻で決まる揺らぎ)。"""
    lag = np.arange(1, n + 1) * 0.045
    pts = _dragon_path(center, t - lag, total, sgn)
    j = np.arange(n) + seed
    jit = np.stack([np.sin(j * 12.9898 + t * 3), np.cos(j * 78.233 + t * 2)], 1) * (8 + lag[:, None] * 60)
    return pts + jit


def _draw_trail(frame, glow, gs, pts):
    """金の光跡(粒)。新しいほど大きく明るい。"""
    n = len(pts)
    for i in range(n):
        a = (1 - i / n) ** 1.2
        c = _scale((90, 215, 255), a)
        if frame is not None:
            cv2.circle(frame, (int(pts[i, 0]), int(pts[i, 1])), max(1, int(3.5 * a + 0.5)), c, -1, AA)
        if glow is not None:
            cv2.circle(glow, (int(pts[i, 0] * gs), int(pts[i, 1] * gs)), max(1, int(9 * a * gs + 0.5)), _scale(c, 0.8),
                       -1)


def _draw_lightning(frame, glow, gs, center, t, seed, size=(1280, 720)):
    """発動中に何度か走る稲妻(円の外、画面の縁からボードの縁へ)。"""
    w, h = size
    for b, (t0, dur) in enumerate(((0.28, 0.16), (0.95, 0.13), (1.7, 0.15), (2.4, 0.12))):
        if not t0 <= t < t0 + dur:
            continue
        a = 1.0 if t < t0 + dur * 0.5 else 0.85 if int(t * 40) % 2 else 0.5     # ちらつき
        rng = np.random.default_rng(seed + b)
        ang = rng.uniform(0, 2 * math.pi)
        end = np.array([center[0] + math.cos(ang) * (QUIET_R + 20), center[1] + math.sin(ang) * (QUIET_R + 20)])
        start = np.array([center[0] + math.cos(ang) * 900, center[1] + math.sin(ang) * 900])
        start = np.clip(start, (0, 0), (w - 1, h - 1))
        n = 11
        s = np.linspace(0, 1, n)[:, None]
        path = start + (end - start) * s
        perp = np.array([-(end - start)[1], (end - start)[0]]) / max(1.0, np.hypot(*(end - start)))
        off = rng.normal(0, 18, n)
        off[0] = off[-1] = 0
        path = path + perp[None, :] * off[:, None]
        bolts = [path]
        for _ in range(2):                             # 枝
            k0 = rng.integers(2, n - 3)
            br = [path[k0]]
            d = (end - start) / n * 0.8 + perp * rng.normal(0, 25)
            for _ in range(4):
                br.append(br[-1] + d + perp * rng.normal(0, 12))
            bolts.append(np.array(br))
        for bp in bolts:
            if glow is not None:
                cv2.polylines(glow, [_i32(bp * gs)], False, _scale((255, 160, 200), a), 3)
            if frame is not None:
                cv2.polylines(frame, [_i32(bp)], False, _scale((255, 140, 190), a), 5, AA)
                cv2.polylines(frame, [_i32(bp)], False, _scale((255, 245, 250), a), 2, AA)


_CLOUDS = ((-1, 175, 625, 1.0), (1, 1125, 95, 0.85))       # 右下はランプがあるので置かない


def _cloud_sprite(d, s_):
    """瑞雲のスプライト(BGR, mask)。金の縁取りの塊 + 渦巻き + たなびく尾。"""
    key = ("cloud", d, s_)
    if key in _glyph_cache:
        return _glyph_cache[key]
    W, H = int(260 * s_), int(110 * s_)
    img = np.zeros((H, W, 3), np.uint8)
    m = np.zeros((H, W), np.uint8)
    x, y = W / 2 - 20 * s_ * d, H / 2
    lumps = [(-48, 6, 20), (-22, -10, 26), (8, -14, 30), (38, -4, 24), (60, 8, 16), (14, 10, 24), (-20, 12, 20)]
    for dx, dy, r in lumps:
        c = (int(x + dx * s_ * d), int(y + dy * s_))
        cv2.circle(img, c, int((r + 3) * s_), D_GOLD, -1, AA)
        cv2.circle(m, c, int((r + 3) * s_), 255, -1, AA)
    for dx, dy, r in lumps:
        cv2.circle(img, (int(x + dx * s_ * d), int(y + dy * s_)), int(r * s_), (70, 35, 120), -1, AA)
    for i, (dx, dy, r) in enumerate(lumps[:5]):
        c = (int(x + dx * s_ * d), int(y + dy * s_))
        rr = int(r * s_ * 0.55)
        cv2.ellipse(img, c, (rr, rr), (i * 70) % 360, 0, 270, D_GOLD, 2, AA)
        cv2.ellipse(img, c, (rr // 2, rr // 2), (i * 70 + 90) % 360, 0, 240, D_GOLD, 2, AA)
    tail = np.array([(int(x + (60 + k * 14) * s_ * d), int(y + (12 + 6 * math.sin(k * 0.9)) * s_)) for k in range(7)],
                    np.int32)
    cv2.polylines(img, [tail], False, D_GOLD, 3, AA)
    cv2.polylines(m, [tail], False, 255, 3, AA)
    _glyph_cache[key] = (img, (m > 100).astype(np.uint8) * 255)
    return _glyph_cache[key]


def _draw_clouds(frame, t, a, shake):
    """瑞雲が隅をゆっくり流れる(スプライトを動かすだけ)。"""
    if a <= 0:
        return
    H, W = frame.shape[:2]
    for i, (d, cx, cy, s_) in enumerate(_CLOUDS):
        img, m = _cloud_sprite(d, s_)
        sh, sw = m.shape
        x0 = int(cx + d * 30 * math.sin(t * 0.7 + i) + shake[0] - sw / 2)
        y0 = int(cy + 5 * math.sin(t * 1.3 + i * 2) + shake[1] - sh / 2)
        fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + sw), min(H, y0 + sh)
        if fx1 <= fx0 or fy1 <= fy0:
            continue
        sub = (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))
        roi = frame[fy0:fy1, fx0:fx1]
        if a >= 1:
            cv2.copyTo(img[sub], m[sub], roi)
        else:
            mixed = cv2.addWeighted(img[sub], a, roi, 1 - a, 0)
            cv2.copyTo(mixed, m[sub], roi)


def _seal_image():
    """判子風の「龍」の印(赤地に白抜き、少し傾けてかすれ付き)。(BGR, alpha) を返す。"""
    if "seal" in _glyph_cache:
        return _glyph_cache["seal"]
    S = 128
    img = np.zeros((S, S, 3), np.uint8)
    al = np.zeros((S, S), np.uint8)
    cv2.rectangle(al, (8, 8), (S - 9, S - 9), 255, -1)
    img[:] = (40, 30, 195)
    cv2.rectangle(img, (15, 15), (S - 16, S - 16), (235, 240, 250), 3)
    m = _ff_mask("龍", 104)
    if m is None:
        m = _hershey_mask("x2", 3.0, 7)
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    sc = min(82 / m.shape[0], 82 / m.shape[1])
    m = cv2.resize(m, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    y0, x0 = (S - m.shape[0]) // 2, (S - m.shape[1]) // 2
    img[y0:y0 + m.shape[0], x0:x0 + m.shape[1]][m > 110] = (235, 240, 250)
    rng = np.random.default_rng(5)                     # かすれ(縁の近くだけ)
    for _ in range(45):
        side = rng.integers(0, 4)
        u = int(rng.integers(8, S - 8))
        v = int(rng.integers(8, 19))
        x, y = [(u, v), (S - v, u), (u, S - v), (v, u)][side]
        cv2.circle(al, (x, y), int(rng.integers(1, 3)), 0, -1)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), 9, 1.0)
    img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR)
    al = cv2.warpAffine(al, M, (S, S), flags=cv2.INTER_LINEAR)
    _glyph_cache["seal"] = (img, al)
    return img, al


def _draw_seal(frame, t, shake):
    """「龍」の印がドーンと押される(t=0.55)。"""
    t0 = 0.55
    if t < t0 or t > 2.6:
        return
    a = 1.0 if t < 2.2 else max(0.0, (2.6 - t) / 0.4)
    tau = t - t0
    sc = 1 + 1.4 * math.exp(-tau * 16)
    img, al = _seal_image()
    sz = int(img.shape[0] * sc)
    img = cv2.resize(img, (sz, sz), interpolation=cv2.INTER_LINEAR)
    al = cv2.resize(al, (sz, sz), interpolation=cv2.INTER_LINEAR)
    cx, cy = 150 + shake[0], 140 + shake[1]
    x0, y0 = int(cx - sz / 2), int(cy - sz / 2)
    H, W = frame.shape[:2]
    fx0, fy0, fx1, fy1 = max(0, x0), max(0, y0), min(W, x0 + sz), min(H, y0 + sz)
    if fx1 <= fx0 or fy1 <= fy0:
        return
    sub = (slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))
    roi = frame[fy0:fy1, fx0:fx1]
    m = cv2.threshold(al[sub], 127, 255, cv2.THRESH_BINARY)[1]
    a *= min(1.0, tau / 0.04)
    src = img[sub] if a >= 1 else cv2.addWeighted(img[sub], a, roi, 1 - a, 0)
    cv2.copyTo(np.ascontiguousarray(src), m, roi)


def _shake(t):
    """印が押された瞬間の、外周要素の揺れ。"""
    tau = t - 0.55
    if tau < 0:
        return (0, 0)
    e = 10 * math.exp(-tau * 7)
    return (int(e * math.sin(tau * 61)), int(e * math.cos(tau * 47)))


def _draw_dragons(frame, center, t, total, intro, seed, glow):
    """2匹の龍(赤龍が宝珠を追い、金龍が絡み合うように続く)、光跡、ブレス、稲妻。
    先に低解像度のグロー(輪郭の光・宝珠・ブレス・稲妻)を加算してから、本体を上に描く。"""
    h, w = frame.shape[:2]
    gs = glow.shape[1] / w
    if not 0 <= t < total:
        return
    red = _dragon_geom(center, t, total, 1, 1.0)
    gold = _dragon_geom(center, t - 0.2, total, -1, 0.82)
    pearl = _dragon_path(center, np.array([t + 0.16]), total, 0)[0]
    breath = intro and 0.35 < t < 2.2
    kb = min(1.0, (t - 0.35) / 0.15) * min(1.0, (2.2 - t) / 0.2) if breath else 0.0
    for (P, tang, nrm, width), pal in ((gold, PAL_GOLD), (red, PAL_RED)):
        Wn = nrm * (width * 1.35)[:, None]
        poly = np.concatenate([P[:DRAGON_N] + Wn, (P[:DRAGON_N] - Wn)[::-1]]) * gs
        cv2.fillPoly(glow, [_i32(poly)], pal["glow"])
    n_r, n_g = (12, 8) if intro else (9, 6)
    tr_r = _trail_points(center, t, total, 1, n_r, 0)
    tr_g = _trail_points(center, t - 0.2, total, -1, n_g, 50)
    _draw_trail(None, glow, gs, tr_r)
    _draw_trail(None, glow, gs, tr_g)
    _draw_pearl(None, glow, gs, pearl, t)
    if breath:
        P, tang, nrm, width = red
        _draw_breath(None, glow, gs, _mouth(P, tang, nrm, 1.75), _breath_dir(tang, nrm), t, kb)
    if intro:
        _draw_lightning(None, glow, gs, center, t, seed, (w, h))
    glow = cv2.GaussianBlur(glow, (0, 0), 2.2)
    bx, by, bw, bh = cv2.boundingRect(cv2.cvtColor(glow, cv2.COLOR_BGR2GRAY))   # 光っている範囲だけ拡大・加算
    if bw and bh:
        X0, Y0 = int(bx / gs), int(by / gs)
        X1, Y1 = min(w, int((bx + bw) / gs)), min(h, int((by + bh) / gs))
        up = cv2.resize(glow[by:by + bh, bx:bx + bw], (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_LINEAR)
        roi = frame[Y0:Y1, X0:X1]
        cv2.addWeighted(roi, 1.0, up, 0.45, 0, dst=roi)
    _draw_trail(frame, None, gs, tr_r)
    _draw_trail(frame, None, gs, tr_g)
    _draw_dragon_body(frame, *gold, t, PAL_GOLD, 1.35, legs=False, detail=False)
    _draw_dragon_body(frame, *red, t, PAL_RED, 1.75)
    _draw_pearl(frame, None, gs, pearl, t)
    if breath:
        P, tang, nrm, width = red
        _draw_breath(frame, None, gs, _mouth(P, tang, nrm, 1.75), _breath_dir(tang, nrm), t, kb)
    if intro:
        _draw_lightning(frame, None, gs, center, t, seed, (w, h))


def _fire_cover(t, total):
    """炎のリングの点火範囲: ブレスが当たった所(頭の少し先)まで。2.0〜2.6 秒で残りも燃え広がる。"""
    a0 = _dragon_phase(0.35, total)[0] + 0.3
    a1 = _dragon_phase(min(t, 2.2), total)[0] + 0.3 if t > 0.35 else a0
    fill = min(1.0, max(0.0, (t - 2.0) / 0.6))
    return a0, a1, fill


def _bonus_level(now, bonus_start, bonus_until):
    """炎の強さ。立ち上がり 0.4 秒、終了 FIRE_FADE 秒前から弱まって明滅。"""
    k = min(1.0, (now - bonus_start) / 0.4)
    rem = bonus_until - now
    if rem < FIRE_FADE:
        k *= (0.35 + 0.65 * rem / FIRE_FADE) * (1.0 if math.sin(now * 38) > -0.2 else 0.25)
    return max(0.0, k)


def _draw_fire_ring(frame, center, now, k, cover=None, glow=None, gs=0.25):
    """的の外周を囲む炎のリング(ノイズで揺らぐ炎の舌 + 火の粉 + 弱いグロー)。半分の解像度で描いて拡大・加算。"""
    if k <= 0:
        return
    cx, cy = center
    H, W = frame.shape[:2]
    RO = QUIET_R + 95
    x0, y0, x1, y1 = max(0, cx - RO), max(0, cy - RO), min(W, cx + RO), min(H, cy + RO)
    x0, y0 = x0 + (x1 - x0) % 2, y0 + (y1 - y0) % 2
    S = 0.5
    lay = np.zeros(((y1 - y0) // 2, (x1 - x0) // 2, 3), np.uint8)
    ox, oy = (cx - x0) * S, (cy - y0) * S
    M = 40
    r0 = (QUIET_R + 2) * S
    layers = (((20, 30, 170), 1.0, 0.8), ((10, 110, 250), 0.66, 0.58), ((120, 230, 255), 0.34, 0.36))
    i = np.arange(M)
    th = i * 2 * math.pi / M + 0.03 * np.sin(now * 3 + i)
    L = 34 + 26 * (0.5 * np.sin(now * 7.3 + i * 1.7) + 0.3 * np.sin(now * 11.9 + i * 0.83)
                   + 0.2 * np.sin(now * 4.1 + i * 2.9))
    L = L * (0.55 + 0.45 * k) * S
    if cover is not None:          # 点火済みの角度だけ燃やす(点火の先端は短い炎から伸びる)
        a0, a1, fill = cover
        d = (th - a0) % (2 * math.pi)
        lit = np.clip((a1 - a0 - d) / 0.5, 0, 1) if a1 > a0 else np.zeros(M)
        if a1 - a0 >= 2 * math.pi:
            lit[:] = 1
        lit = np.maximum(lit, fill)
        L = L * lit
        keep = lit > 0.03
    else:
        keep = np.ones(M, bool)
    tip_th = th + 0.11 * np.sin(now * 9 + i * 1.3) + 0.05 * np.sin(now * 17 + i * 0.7)
    for col, sl, sw in layers:              # 赤(長い) → 橙 → 黄(短い芯) の順に重ねる
        hw = sw * 2 * math.pi / M
        tip_r = r0 + L * sl
        tri = np.stack([np.stack([ox + np.cos(th - hw) * r0, oy + np.sin(th - hw) * r0], 1),
                        np.stack([ox + np.cos(tip_th) * tip_r, oy + np.sin(tip_th) * tip_r], 1),
                        np.stack([ox + np.cos(th + hw) * r0, oy + np.sin(th + hw) * r0], 1)], 1)
        tri = np.round(tri).astype(np.int32)[keep]
        for par in (0, 1):   # 隣どうしは重なるので偶奇に分ける(fillPoly は重なりを穴にする)
            cv2.fillPoly(lay, list(tri[par::2]), _scale(col, 0.65 * k), AA)
    # 火の粉(状態なし: 時刻から位置を決める)
    for i in range(28):
        rate = 0.9 + (i * 37 % 11) / 10
        f = (now * rate + i * 0.618) % 1.0
        th = i * 2.399 + 0.25 * math.sin(now * 2 + i)
        if cover is not None:
            a0, a1, fill = cover
            if fill < 0.5 and (th - a0) % (2 * math.pi) > a1 - a0:
                continue
        r = r0 + (20 + f * 90) * S
        col = (60, int(200 - 140 * f), 255) if f < 0.5 else (20, int(120 - 100 * f), int(255 - 120 * f))
        cv2.circle(lay, (int(ox + math.cos(th) * r), int(oy + math.sin(th) * r)), max(1, int(2 * (1 - f))),
                   _scale(col, k), -1, AA)
    if glow is not None:       # 龍と共通のグロー層へ(まとめてぼかして加算する)
        gx0, gy0 = int(x0 * gs), int(y0 * gs)
        gw, gh = min(glow.shape[1] - gx0, int((x1 - x0) * gs)), min(glow.shape[0] - gy0, int((y1 - y0) * gs))
        sub = glow[gy0:gy0 + gh, gx0:gx0 + gw]
        cv2.add(sub, cv2.resize(lay, (gw, gh), interpolation=cv2.INTER_AREA), dst=sub)
    else:
        g = cv2.GaussianBlur(cv2.resize(lay, (lay.shape[1] // 2, lay.shape[0] // 2), interpolation=cv2.INTER_AREA),
                             (0, 0), 3)
        lay = cv2.addWeighted(lay, 1.0, cv2.resize(g, (lay.shape[1], lay.shape[0]), interpolation=cv2.INTER_LINEAR),
                              0.6, 0)
    up = cv2.resize(lay, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    roi = frame[y0:y1, x0:x1]
    cv2.add(roi, up, dst=roi)


def _draw_bonus_telop(frame, w, h, center, t, shake=(0, 0)):
    """右側の大テロップ: DOUBLE / ×2 / SCORE x2 5 SEC。"""
    if not 0 <= t < DRAGON_T + 0.4:
        return
    a = 1.0 if t < DRAGON_T - 0.1 else max(0.0, (DRAGON_T + 0.4 - t) / 0.5)
    cx = w - 118 + shake[0]   # 龍の軌道(外周 r≈390 + 頭)と重ならないよう右端寄せ
    pop = 1 + 0.8 * math.exp(-t * 9) * abs(math.cos(t * 15))
    sy = shake[1]
    _text_center(frame, "DOUBLE", cx, h * 0.32 + sy, DUPLEX, 1.3, _scale(INK, a), 9)
    _text_center(frame, "DOUBLE", cx, h * 0.32 + sy, DUPLEX, 1.3, _scale(D_GOLD, a), 3)
    m, e = _small_mask("×2", 130)
    rows = np.arange(m.shape[0], dtype=np.float32) / m.shape[0]
    grad = np.stack([20 + 25 * (1 - rows), 50 + 190 * (1 - rows), 240 + 15 * (1 - rows)], 1)    # 黄→赤
    flick = 0.92 + 0.08 * math.sin(t * 30)
    fill = (grad[:, None, :] * a * flick).clip(0, 255).astype(np.uint8)
    _paste_mask(frame, m, e, cx + 4 * math.sin(t * 37), h * 0.5 + sy, pop, fill, (0, 0, 0))
    _text_center(frame, "SCORE x2  5 SEC", cx, h * 0.68 + sy, DUPLEX, 0.58, _scale(INK, a), 6)
    _text_center(frame, "SCORE x2  5 SEC", cx, h * 0.68 + sy, DUPLEX, 0.58, _scale((235, 235, 235), a), 1)


def _draw_bonus_bar(frame, w, h, center, radii, now, bonus_start, bonus_until):
    """スコアバーの ×2 バッジと残り秒数のゲージ。"""
    bx0, by0, bx1, _ = _bar_rect(w, h, center, radii)
    rem = max(0.0, bonus_until - now)
    total = max(1e-6, bonus_until - bonus_start)
    blink = rem < FIRE_FADE and math.sin(now * 38) < -0.2
    cv2.rectangle(frame, (bx0 + 66, by0 + 6), (bx0 + 98, by0 + 24), (40, 40, 60) if blink else (0, 60, 230), -1)
    cv2.putText(frame, "x2", (bx0 + 71, by0 + 21), DUPLEX, 0.5, WHITE, 1, AA)
    cv2.putText(frame, f"{rem:.1f}s", (bx0 + 104, by0 + 21), SIMPLEX, 0.4, (120, 190, 255), 1, AA)
    gx0, gx1, gy = bx0 + 4, bx1 - 4, by0 + 77
    cv2.line(frame, (gx0, gy), (gx1, gy), (50, 50, 55), 3)
    cv2.line(frame, (gx0, gy), (gx0 + int((gx1 - gx0) * rem / total), gy), (0, 130, 255), 3)


# ---------------------------------------------------------------- lamp (ブルで「ペカッ」と光るランプ)
# 右下に常設のオリジナルのランプ(丸い筐体 + ガラスに「BULL!」)。消灯時は静的画像に含めて静止。
# ブル命中で 0.05 秒で点灯 → ゆっくり明滅 → 演出の終わりにふっと消える。背景(半径300の外)もランプの光で色づく。
LAMP_W, LAMP_H = 150, 92
LAMP_TEXT = "BULL!"
PEKA_PINK = (170, 90, 255)
PEKA_YELLOW = (70, 225, 255)


def _lamp_center(w, h):
    return w - 142, h - 132


def _lamp_glass_mask(w, h):
    """ガラス部分(角の丸いドーム)の 1ch マスクと、その外接矩形。"""
    key = ("lampmask", w, h)
    if key not in _glyph_cache:
        cx, cy = _lamp_center(w, h)
        x0, y0 = cx - LAMP_W // 2, cy - LAMP_H // 2
        m = np.zeros((LAMP_H, LAMP_W), np.uint8)
        r = LAMP_H // 2
        cv2.ellipse(m, (LAMP_W // 2, r), (LAMP_W // 2 - 1, r - 1), 0, 180, 360, 255, -1, AA)   # 上はドーム
        cv2.rectangle(m, (1, r), (LAMP_W - 2, LAMP_H - 1), 255, -1)
        _glyph_cache[key] = (m, (x0, y0))
    return _glyph_cache[key]


def _draw_lamp_body(img, w, h, glass, text_col, edge_col=(90, 90, 96)):
    """ランプ一式(台座・ガラス・文字)。glass は BGR か (LAMP_H, LAMP_W, 3) の色画像。"""
    cx, cy = _lamp_center(w, h)
    m, (x0, y0) = _lamp_glass_mask(w, h)
    # 台座(クロームの帯)
    cv2.rectangle(img, (x0 - 8, y0 + LAMP_H - 4), (x0 + LAMP_W + 7, y0 + LAMP_H + 14), (52, 52, 56), -1)
    cv2.rectangle(img, (x0 - 8, y0 + LAMP_H - 4), (x0 + LAMP_W + 7, y0 + LAMP_H - 1), (120, 120, 128), -1)
    for k in range(5):                                  # 台座のねじ
        cv2.circle(img, (x0 + 6 + k * (LAMP_W - 12) // 4, y0 + LAMP_H + 7), 2, (95, 95, 100), -1, AA)
    roi = img[y0:y0 + LAMP_H, x0:x0 + LAMP_W]
    src = np.ascontiguousarray(glass) if isinstance(glass, np.ndarray) else np.full_like(roi, glass)
    cv2.copyTo(src, m, roi)
    # 縁取りとガラスの映り込み
    r = LAMP_H // 2
    cv2.ellipse(img, (cx, y0 + r), (LAMP_W // 2, r), 0, 180, 360, edge_col, 2, AA)
    cv2.line(img, (x0, y0 + r), (x0, y0 + LAMP_H - 2), edge_col, 2, AA)
    cv2.line(img, (x0 + LAMP_W - 1, y0 + r), (x0 + LAMP_W - 1, y0 + LAMP_H - 2), edge_col, 2, AA)
    cv2.ellipse(img, (cx - 22, y0 + 22), (34, 9), -12, 200, 330, (200, 200, 205) if isinstance(glass, np.ndarray)
                else (75, 70, 80), 2, AA)
    _text_center(img, LAMP_TEXT, cx, cy + 10, DUPLEX, 1.05, text_col, 2)


def _draw_lamp_off(img, w, h):
    _draw_lamp_body(img, w, h, (40, 30, 46), (78, 62, 92))


def _peka_level(age, end):
    """点灯の強さ。0.05 秒でパッと点き、ゆっくり明滅し、終わりの 0.35 秒でふっと消える。"""
    if age < 0:
        return 0.0
    on = min(1.0, age / 0.05)
    blink = 0.82 + 0.18 * math.cos(2 * math.pi * 1.6 * age)
    out = min(1.0, max(0.0, (end - age) / 0.35))
    return on * blink * out


def _peka_tints(w, h):
    """ランプから背景へ広がる光(ピンク・黄)の加算用画像。サイズごとにキャッシュ。"""
    key = ("peka", w, h)
    if key not in _glyph_cache:
        cx, cy = _lamp_center(w, h)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.hypot(xx - cx, yy - cy)
        f = np.clip(1 - d / 1100, 0, 1) ** 2.2
        ang = np.arctan2(yy - cy, xx - cx)
        rays = 0.55 + 0.45 * np.cos(ang * 14) ** 2                     # 放射状の筋
        out = []
        for col in (PEKA_PINK, PEKA_YELLOW):
            out.append((f[..., None] * np.array(col, np.float32)).astype(np.uint8))
            out.append((f[..., None] * rays[..., None] * np.array(col, np.float32)).astype(np.uint8))
        _glyph_cache[key] = out
    return _glyph_cache[key]


def _draw_peka(frame, w, h, age, end, fever, bg=True):
    """ペカり: 背景の光 → 点灯したランプ → ランプ周りの光の輪とトゲ。"""
    lv = _peka_level(age, end)
    if lv <= 0:
        return
    cx, cy = _lamp_center(w, h)
    if bg:
        pink, pink_r, yel, yel_r = _peka_tints(w, h)
        burst = 1 + 0.3 * max(0.0, 1 - age / 0.25)
        if fever:      # 背景全体: 放射の筋の入った光が、ピンクと黄で交互に脈打つ
            tint = pink_r if int(age * 3.2) % 2 else yel_r
            cv2.addWeighted(frame, 1.0, tint, 0.2 * lv * (1 + 0.5 * (burst - 1)), 0, dst=frame)
        else:          # 控えめ: ほんのり
            cv2.addWeighted(frame, 1.0, pink, 0.08 * lv * burst, 0, dst=frame)
    # ランプ周りの光の輪(点いた瞬間は大きく)。2倍タイム中(bg=False)は龍と炎で明るいので省く
    if bg:
        burst = max(0.0, 1 - age / 0.3)
        halo_r = int(95 + 70 * burst)
        mg = int(halo_r * 1.3) + 30                       # トゲ(1.25倍)とぼかしの裾まで入る大きさ
        x0, y0 = max(0, cx - mg), max(0, cy - mg)
        x1, y1 = min(w, cx + mg), min(h, cy + mg)
        q = 2                                              # 半分の解像度で描いてぼかし、拡大して加算
        lay = np.zeros(((y1 - y0) // q, (x1 - x0) // q, 3), np.uint8)
        col = _hue(age * 160) if fever else PEKA_PINK
        oc = ((cx - x0) // q, (cy - y0) // q)
        cv2.circle(lay, oc, halo_r // q, _scale(col, 0.55 * lv), -1)
        n = 16
        for i in range(n):                               # トゲ(回る光の筋)
            th = i * 2 * math.pi / n + age * (1.2 if fever else 0.5)
            L = halo_r * (1.25 if i % 2 else 1.0) / q
            cv2.line(lay, oc, (int(oc[0] + math.cos(th) * L), int(oc[1] + math.sin(th) * L)),
                     _scale(PEKA_YELLOW if i % 2 else col, 0.8 * lv), 2)
        lay = cv2.resize(cv2.GaussianBlur(lay, (0, 0), 4.5), (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
        roi = frame[y0:y1, x0:x1]
        cv2.add(roi, lay, dst=roi)
    # 点灯したランプ(ガラスはグラデーション、fever は虹色が流れる)
    rows = np.arange(LAMP_H, dtype=np.float32)[:, None]
    cols = np.arange(LAMP_W, dtype=np.float32)[None, :]
    if fever:
        hue = ((rows * 0.9 + cols * 0.5 + age * 220) % 180).astype(np.int32)
        glass = _HUE_LUT[hue].astype(np.float32)
        glass = glass * 0.55 + 255 * 0.45                 # 白っぽく明るいガラス
    else:
        t_ = np.clip(rows / LAMP_H, 0, 1)[..., None]
        glass = np.array(PEKA_YELLOW, np.float32) * (1 - t_) + np.array(PEKA_PINK, np.float32) * t_
        glass = np.broadcast_to(glass, (LAMP_H, LAMP_W, 3)) * 0.8 + 255 * 0.2
    dim = 0.55 + 0.45 * lv
    glass = (glass * dim).clip(0, 255).astype(np.uint8)
    _draw_lamp_body(frame, w, h, glass, (60, 10, 150), (210, 210, 220))


# ---------------------------------------------------------------- main
def render_projector(w: int, h: int, center: tuple[int, int], radii: tuple[int, int, int],
                     total_score: int, hits: list[Hit],
                     cooldown_progress: float | None,
                     calibrating: bool, now: float, mode: str = "normal",
                     bonus_start: float | None = None, bonus_until: float | None = None) -> np.ndarray:
    """投影画面を1フレーム描いて BGR uint8 (h, w, 3) を返す。

    radii は (r100, r50, r10)。hits は古い順の全履歴。
    cooldown_progress はクールダウン中 0→1、READY なら None。
    calibrating=True のときは真っ白(カメラの投影範囲検出用)。
    mode は "normal" か "fever"。fever はブルの演出だけ派手になり、隅に FEVER MODE を出す。
    bonus_start <= now < bonus_until の間は2倍タイム(fever のみ): 発動時に龍、期間中は炎のリング。
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
    fever = mode == "fever"
    bonus = (fever and bonus_start is not None and bonus_until is not None
             and bonus_start <= now < bonus_until)
    intro = bonus and now - bonus_start < DRAGON_T + 0.4
    bull = _last_bull(hits, now, FEVER_END if fever else BULL_END)
    box = _bar_rect(w, h, center, radii)
    outer = bool(bull) or bonus
    if bull:
        hit, age, streak = bull
        if fever:
            # 2倍タイム中は龍が舞っているので、バースト・大当たり文字・紙吹雪を省く(描画時間・輝度対策)
            trig = bonus
            _draw_fever_outer(frame, w, h, center, radii, hit, age, streak, dragon=trig, fire=bonus)
        else:
            _draw_rays(frame, center, age)
            _draw_confetti(frame, center, radii, hit, age)
            _draw_telop(frame, w, h, center, radii, age, streak)
    if bull:    # ペカり(背景の光は 2倍タイム中は省く: 龍と炎で十分明るい)
        _draw_peka(frame, w, h, bull[1], FEVER_END if fever else BULL_END, fever, bg=not bonus)
    if bonus:   # 炎は加算なので、バーストの後に重ねる。龍は bonus_until まで舞い、終了前に去る
        tb, total = now - bonus_start, bonus_until - bonus_start
        glow = np.zeros((h // 4, w // 4, 3), np.uint8)   # 龍・炎・宝珠・稲妻の共通グロー(1/4 解像度)
        _draw_fire_ring(frame, center, now, _bonus_level(now, bonus_start, bonus_until), _fire_cover(tb, total),
                        glow, 0.25)
        shake = _shake(tb) if intro else (0, 0)
        ca = min(1.0, tb / 0.5) * min(1.0, max(0.0, (total - tb) / 0.8))
        _draw_clouds(frame, tb, ca, shake)
        if intro:
            _draw_bonus_telop(frame, w, h, center, tb, shake)
            _draw_seal(frame, tb, shake)
        _draw_dragons(frame, center, tb, total, intro, int(bonus_start * 10) & 0xFFFF, glow)
    if outer:
        frame[box[1]:box[3] + 1, box[0]:box[2] + 1] = base[box[1]:box[3] + 1, box[0]:box[2] + 1]
    reel = bull[1] if (fever and bull and bull[0] is hits[-1]) else None
    _draw_bar(frame, w, h, center, radii, total_score, hits, cooldown_progress, now, fever=reel)
    if bonus:
        _draw_bonus_bar(frame, w, h, center, radii, now, bonus_start, bonus_until)
    if fever:
        _draw_fever_badge(frame, w, h, center, radii)
    if outer:      # 演出の下に隠れないよう、四隅マーカーは最後に描き直す
        _draw_corners(frame, w, h)
    if outer:
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
