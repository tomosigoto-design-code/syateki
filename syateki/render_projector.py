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


def _draw_stars(frame, center, hit, age):
    """星のきらめき(4本のとげの星が瞬く)。円の外の固定位置。"""
    a = _fever_fade(age)
    rng = np.random.default_rng((int(hit.t * 1000) + 7) & 0xFFFFFFFF)
    h, w = frame.shape[:2]
    pts = rng.uniform((0, 0), (w, h), (160, 2))
    d = np.hypot(pts[:, 0] - center[0], pts[:, 1] - center[1])
    pts = pts[d > QUIET_R + 20][:60]
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


def _draw_marquee(frame, age):
    """画面外周の電飾(色が流れて、チカチカ点滅する)。"""
    a = _fever_fade(age)
    h, w = frame.shape[:2]
    m, step = 13, 30
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
    sc = min(1.0, (h - 70) / total, (w - 30 - x_l) / max(m.shape[1] for m, _ in glyphs))
    y = (h - total * sc) / 2
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
    """dragon=True(2倍タイムの発動中)は龍とテロップが主役なので、バーストと大当たり文字を省く。"""
    if not dragon:
        _draw_sunburst(frame, center, age, wedges=not fire)   # 炎と重なるときは帯を省く(平均輝度対策)
    _draw_stars(frame, center, hit, age)
    _draw_confetti(frame, center, radii, hit, age, n=90 if dragon else (140 if fire else 200),
                   palette=RAINBOW, a=_fever_fade(age), seed=11)
    if not dragon:
        _draw_rain(frame, hit, age)
    _draw_marquee(frame, age)
    if not dragon:
        _draw_fever_telops(frame, w, h, center, age, streak)
        _draw_jackpot(frame, w, h, center, age)


# ---------------------------------------------------------------- bonus (fever の2倍タイム: 龍 + 炎のリング)
# bonus_start <= now < bonus_until の間だけ。龍・炎・テロップはすべて半径300の外(描いた後に円内を静的画像へ戻す)。
DRAGON_T = 2.1           # 発動演出(龍)の長さ
DRAGON_R0 = 345          # 龍が回る軌道の半径
DRAGON_N = 46            # 胴の節の数
FIRE_FADE = 1.0          # 終了何秒前から炎が弱まるか
D_RED = (34, 36, 196)
D_RED_DK = (20, 20, 110)
D_RED_HI = (70, 80, 240)
D_GOLD = (50, 185, 245)
D_GOLD_DK = (20, 110, 170)
D_CREAM = (170, 225, 245)


def _dragon_points(center, phi, phi0, t):
    """龍の軌道(phi は配列)。外から渦を巻いて入り、ボードの外周を回って、渦を巻いて去る。"""
    a_in, a_out = phi0 + 0.8, phi0 + 0.8 + 3.7
    r = DRAGON_R0 + 700 * np.maximum(0.0, a_in - phi) + 420 * np.maximum(0.0, phi - a_out) ** 1.3
    r = r + 9 * np.sin(phi * 9 - t * 10)                                  # うねり
    return np.stack([center[0] + np.cos(phi) * r, center[1] + np.sin(phi) * r], 1)


def _draw_dragon(frame, center, t):
    """龍。胴は太さの変わる帯を多角形で塗る(縁取り → 胴 → 背の光沢 → 腹の金 → 鱗)。"""
    if not 0 <= t < DRAGON_T:
        return
    phi0 = math.pi * 0.62
    phi_h = phi0 + 0.15 + 4.15 * t                          # 頭の角度(時計回り)
    N = DRAGON_N
    P = _dragon_points(center, phi_h - np.arange(N + 1) * 0.046, phi0, t)
    tang = P[:-1] - P[1:]
    tang /= np.maximum(1e-6, np.hypot(tang[:, 0], tang[:, 1]))[:, None]
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    flip = np.sign((nrm * (P[:-1] - np.array(center))).sum(1))[:, None]   # 背中は外向き
    nrm = nrm * np.where(flip == 0, 1, flip)
    j = np.arange(N)
    width = 22 * (0.7 + 0.3 * np.sin(np.clip(j / 8, 0, 1) * math.pi / 2)) * (1 - 0.82 * (j / N) ** 1.3)
    Q = P[:N]
    Wn = nrm * width[:, None]
    ip = lambda v: (int(round(v[0])), int(round(v[1])))
    i32 = lambda a: np.round(a).astype(np.int32)

    # 背びれ(のこぎり状のたてがみ): 外側の暗い三角 + 内側の色付き三角
    ks = np.arange(2, N - 1, 2)
    fl = width[ks] * 0.8 + 5 + 3 * np.sin(t * 12 + ks)
    T, Nn = tang[ks], nrm[ks]
    base = Q[ks] + Nn * (width[ks] * 0.5)[:, None]
    tip = Q[ks] + Nn * (width[ks] + fl)[:, None] - T * 14
    fins = np.stack([base + T * 9, tip, base - T * 11], 1)
    cv2.fillPoly(frame, list(i32(fins)), D_RED_DK, AA)
    inner = np.stack([fins[:, 0], (fins[:, 0] + fins[:, 1]) / 2, fins[:, 2]], 1)
    gold = ks % 4 == 0
    cv2.fillPoly(frame, list(i32(inner[gold])), D_GOLD, AA)
    cv2.fillPoly(frame, list(i32(inner[~gold])), D_RED, AA)
    # 脚(腹側に2対、金の爪)
    for k in (11, 29):
        (tx, ty), (nx, ny), (x, y), w_ = tang[k], nrm[k], Q[k], width[k]
        for d in (-5, 5):
            sw = 6 * math.sin(t * 14 + k + d)
            b = (x - nx * w_ * 0.6 + tx * d, y - ny * w_ * 0.6 + ty * d)
            kn = (b[0] - nx * 16 + tx * (sw - 6), b[1] - ny * 16 + ty * (sw - 6))
            ft = (kn[0] - nx * 4 + tx * 10, kn[1] - ny * 4 + ty * 10)
            cv2.line(frame, ip(b), ip(kn), D_RED_DK, 7, AA)
            cv2.line(frame, ip(b), ip(kn), D_RED, 4, AA)
            cv2.line(frame, ip(kn), ip(ft), D_RED, 4, AA)
            for c_ in (-1, 0, 1):
                cl = (ft[0] - nx * (5 - c_ * 2) + tx * (4 + c_ * 3), ft[1] - ny * (5 - c_ * 2) + ty * (4 + c_ * 3))
                cv2.line(frame, ip(ft), ip(cl), D_CREAM, 2, AA)

    def band(off, half, col, outline=None):
        c_ = Q + Wn * off
        poly = np.concatenate([c_ + Wn * half, (c_ - Wn * half)[::-1]])
        cv2.fillPoly(frame, [i32(poly)], col, AA)
        if outline is not None:
            cv2.polylines(frame, [i32(poly)], True, outline, 2, AA)

    band(0, 1.0, D_RED, D_RED_DK)
    band(0.38, 0.42, D_RED_HI)
    band(-0.58, 0.36, D_GOLD)
    # 鱗(背中側の弧)と腹の節
    kk = np.arange(1, N - 1, 2)
    arc = np.linspace(-0.9, 0.9, 5)
    c_ = Q[kk] + Wn[kk] * 0.15
    r_ = (width[kk] * 0.55)[:, None, None]
    T, Nn = tang[kk][:, None, :], nrm[kk][:, None, :]
    scales = c_[:, None, :] + (-T * np.cos(arc)[None, :, None] + Nn * np.sin(arc)[None, :, None]) * r_
    cv2.polylines(frame, list(i32(scales)), False, D_GOLD_DK, 1, AA)
    bc = Q[kk] - Wn[kk] * 0.58
    plates = np.stack([bc + Wn[kk] * 0.33, bc - Wn[kk] * 0.33], 1)
    cv2.polylines(frame, list(i32(plates)), False, D_GOLD_DK, 1, AA)
    # 尾の房(炎のような毛)
    xt, yt = P[N - 1]
    tx, ty = tang[N - 2]
    nx, ny = nrm[N - 2]
    for s_ in (-1, 0, 1):
        sw = 6 * math.sin(t * 13 + s_)
        tip = (xt - tx * 34 + nx * (s_ * 14 + sw), yt - ty * 34 + ny * (s_ * 14 + sw))
        cv2.fillConvexPoly(frame, np.array([[xt + nx * 5, yt + ny * 5], tip, [xt - nx * 5, yt - ny * 5]], np.int32),
                           D_GOLD if s_ else D_RED, AA)
    _draw_dragon_head(frame, P[0] + tang[0] * 6, tang[0], nrm[0], t)


def _draw_dragon_head(frame, p, f, n, t):
    x, y = p
    fx, fy = f
    nx, ny = n
    HS = 1.6

    def q(a, b):                                       # 頭の座標系 (前方 a, 背側 b)
        return (int(x + (fx * a + nx * b) * HS), int(y + (fy * a + ny * b) * HS))

    def e(a, b, ax, ay, a0, a1, col, th=-1):
        cv2.ellipse(frame, q(a, b), (int(ax * HS), int(ay * HS)), ang, a0, a1, col, th, AA)

    ang = math.degrees(math.atan2(fy, fx))
    # たてがみ(後ろへ流れる)
    for i in range(6):
        sway = 5 * math.sin(t * 14 + i)
        poly = np.array([q(-2 - i * 6, 12 - i), q(-30 - i * 7, 26 + i * 2 + sway), q(-14 - i * 6, 2)], np.int32)
        cv2.fillConvexPoly(frame, poly, D_RED_DK if i % 2 else D_RED, AA)
        cv2.fillConvexPoly(frame, np.array([q(-6 - i * 6, 9), q(-24 - i * 7, 20 + i * 2 + sway), q(-14 - i * 6, 4)],
                                           np.int32), D_GOLD, AA)
    # 角(2本、後ろへ反る)
    for off, col in ((0, D_CREAM), (8, D_GOLD)):
        horn = [q(-2 - off - k * 6, 15 + k * 4.5 - k * k * 0.55) for k in range(7)]
        for k in range(6):
            cv2.line(frame, horn[k], horn[k + 1], col, max(2, int((6 - k) * HS * 0.8)), AA)
    # 下あご(口を開け閉め)・舌・牙
    gape = 7 + 5 * math.sin(t * 12)
    cv2.fillConvexPoly(frame, np.array([q(0, -6), q(40, -12 - gape), q(36, -2), q(10, 0)], np.int32), (25, 15, 100), AA)
    cv2.fillConvexPoly(frame, np.array([q(-2, -6), q(38, -14 - gape), q(35, -21 - gape), q(2, -17)], np.int32), D_RED_DK, AA)
    cv2.fillConvexPoly(frame, np.array([q(-1, -8), q(36, -15 - gape), q(33, -19 - gape), q(3, -15)], np.int32), D_RED, AA)
    cv2.line(frame, q(14, -5 - gape * 0.3), q(46, -8 - gape * 0.4), (60, 60, 230), max(2, int(2 * HS)), AA)
    for k in range(3):
        cv2.fillConvexPoly(frame, np.array([q(16 + k * 7, -2), q(18 + k * 7, -8), q(21 + k * 7, -2)], np.int32),
                           (235, 240, 245), AA)
    cv2.fillConvexPoly(frame, np.array([q(30, -14 - gape), q(32, -8 - gape), q(34, -15 - gape)], np.int32),
                       (235, 240, 245), AA)
    # 頭蓋・鼻づら・ほお
    e(0, 2, 24, 17, 0, 360, D_RED_DK)
    e(0, 2, 22, 15, 0, 360, D_RED)
    e(-2, 7, 14, 7, 0, 360, D_RED_HI)
    e(25, 1, 20, 10, 0, 360, D_RED_DK)
    e(25, 1, 19, 9, 0, 360, D_GOLD)
    e(26, 3, 15, 5, 0, 360, D_CREAM)
    cv2.circle(frame, q(41, 5), max(2, int(2 * HS)), (20, 20, 40), -1, AA)  # 鼻の穴
    # 眉と目(金の目、縦長の瞳)
    e(4, 10, 12, 6, 180, 360, D_GOLD, max(2, int(3 * HS)))
    cv2.circle(frame, q(5, 7), int(6 * HS), (20, 20, 40), -1, AA)
    cv2.circle(frame, q(5, 7), int(5 * HS), (40, 215, 250), -1, AA)
    cv2.ellipse(frame, q(6, 7), (max(1, int(1.3 * HS)), int(4.5 * HS)), ang + 90, 0, 360, (10, 10, 10), -1, AA)
    cv2.circle(frame, q(3, 9), max(1, int(1.2 * HS)), (255, 255, 255), -1, AA)
    # ひげ(長く、なびく)
    for side in (1, -1):              # 鼻先から前へ出て、外側へ大きく弧を描いて後ろへ流れる
        wh = [q(42 + 10 * math.sin(k * 0.35) - k * 4.5, side * (4 + k * 2.6) + 5 * math.sin(k * 0.7 - t * 11 + side))
              for k in range(13)]
        cv2.polylines(frame, [np.array(wh, np.int32)], False, D_GOLD, 2, AA)


def _bonus_level(now, bonus_start, bonus_until):
    """炎の強さ。立ち上がり 0.4 秒、終了 FIRE_FADE 秒前から弱まって明滅。"""
    k = min(1.0, (now - bonus_start) / 0.4)
    rem = bonus_until - now
    if rem < FIRE_FADE:
        k *= (0.35 + 0.65 * rem / FIRE_FADE) * (1.0 if math.sin(now * 38) > -0.2 else 0.25)
    return max(0.0, k)


def _draw_fire_ring(frame, center, now, k):
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
    M = 46
    r0 = (QUIET_R + 2) * S
    layers = (((20, 30, 170), 1.0, 0.8), ((10, 110, 250), 0.66, 0.58), ((120, 230, 255), 0.34, 0.36))
    for i in range(M):
        th = i * 2 * math.pi / M + 0.03 * math.sin(now * 3 + i)
        L = 34 + 26 * (0.5 * math.sin(now * 7.3 + i * 1.7) + 0.3 * math.sin(now * 11.9 + i * 0.83)
                       + 0.2 * math.sin(now * 4.1 + i * 2.9))
        L *= (0.55 + 0.45 * k) * S
        sway = 0.11 * math.sin(now * 9 + i * 1.3) + 0.05 * math.sin(now * 17 + i * 0.7)
        c0, s0 = math.cos(th), math.sin(th)
        ct, st_ = math.cos(th + sway), math.sin(th + sway)
        for col, sl, sw in layers:
            hw = sw * 2 * math.pi / M
            tip_r = r0 + L * sl
            pts = np.array([[ox + math.cos(th - hw) * r0, oy + math.sin(th - hw) * r0],
                            [ox + ct * tip_r, oy + st_ * tip_r],
                            [ox + math.cos(th + hw) * r0, oy + math.sin(th + hw) * r0]], np.int32)
            cv2.fillConvexPoly(lay, pts, _scale(col, 0.82 * k), AA)
    # 火の粉(状態なし: 時刻から位置を決める)
    for i in range(40):
        rate = 0.9 + (i * 37 % 11) / 10
        f = (now * rate + i * 0.618) % 1.0
        th = i * 2.399 + 0.25 * math.sin(now * 2 + i)
        r = r0 + (20 + f * 90) * S
        col = (60, int(200 - 140 * f), 255) if f < 0.5 else (20, int(120 - 100 * f), int(255 - 120 * f))
        cv2.circle(lay, (int(ox + math.cos(th) * r), int(oy + math.sin(th) * r)), max(1, int(2 * (1 - f))),
                   _scale(col, k), -1, AA)
    glow = cv2.GaussianBlur(cv2.resize(lay, (lay.shape[1] // 2, lay.shape[0] // 2), interpolation=cv2.INTER_AREA),
                            (0, 0), 3)
    lay = cv2.addWeighted(lay, 1.0, cv2.resize(glow, (lay.shape[1], lay.shape[0]), interpolation=cv2.INTER_LINEAR),
                          0.6, 0)
    up = cv2.resize(lay, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    roi = frame[y0:y1, x0:x1]
    cv2.add(roi, up, dst=roi)


def _draw_bonus_telop(frame, w, h, center, t):
    """右側の大テロップ: DOUBLE / ×2 / SCORE x2 5 SEC。"""
    if not 0 <= t < DRAGON_T + 0.4:
        return
    a = 1.0 if t < DRAGON_T - 0.1 else max(0.0, (DRAGON_T + 0.4 - t) / 0.5)
    cx = w - 118          # 龍の軌道(外周 r≈390 + 頭)と重ならないよう右端寄せ
    pop = 1 + 0.8 * math.exp(-t * 9) * abs(math.cos(t * 15))
    _text_center(frame, "DOUBLE", cx, h * 0.32, DUPLEX, 1.3, _scale(INK, a), 9)
    _text_center(frame, "DOUBLE", cx, h * 0.32, DUPLEX, 1.3, _scale(D_GOLD, a), 3)
    m, e = _small_mask("×2", 130)
    rows = np.arange(m.shape[0], dtype=np.float32) / m.shape[0]
    grad = np.stack([20 + 25 * (1 - rows), 50 + 190 * (1 - rows), 240 + 15 * (1 - rows)], 1)    # 黄→赤
    flick = 0.92 + 0.08 * math.sin(t * 30)
    fill = (grad[:, None, :] * a * flick).clip(0, 255).astype(np.uint8)
    _paste_mask(frame, m, e, cx + 4 * math.sin(t * 37), h * 0.5, pop, fill, (0, 0, 0))
    _text_center(frame, "SCORE x2  5 SEC", cx, h * 0.68, DUPLEX, 0.58, _scale(INK, a), 6)
    _text_center(frame, "SCORE x2  5 SEC", cx, h * 0.68, DUPLEX, 0.58, _scale((235, 235, 235), a), 1)


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
            # 2倍タイムを発動させたブルは、龍とテロップが主役(バーストと大当たり文字は出さない)
            trig = bonus and abs(hit.t - bonus_start) < 0.5
            _draw_fever_outer(frame, w, h, center, radii, hit, age, streak, dragon=trig, fire=bonus)
        else:
            _draw_rays(frame, center, age)
            _draw_confetti(frame, center, radii, hit, age)
            _draw_telop(frame, w, h, center, radii, age, streak)
    if bonus:   # 炎は加算なので、バーストの後に重ねる
        _draw_fire_ring(frame, center, now, _bonus_level(now, bonus_start, bonus_until))
    if intro:
        _draw_bonus_telop(frame, w, h, center, now - bonus_start)
        _draw_dragon(frame, center, now - bonus_start)
    if outer:
        frame[box[1]:box[3] + 1, box[0]:box[2] + 1] = base[box[1]:box[3] + 1, box[0]:box[2] + 1]
    reel = bull[1] if (fever and bull and bull[0] is hits[-1]) else None
    _draw_bar(frame, w, h, center, radii, total_score, hits, cooldown_progress, now, fever=reel)
    if bonus:
        _draw_bonus_bar(frame, w, h, center, radii, now, bonus_start, bonus_until)
    if fever:
        _draw_fever_badge(frame, w, h, center, radii)
        if outer:
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
