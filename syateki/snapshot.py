"""ウィンドウを開かずに画面をPNG出力し、描画時間を計る。

    python -m syateki.snapshot <outdir>
"""
import os
import sys
import time

import cv2
import numpy as np

from . import config
from .game import GameState
from .render_monitor import render_monitor
from .render_projector import render_projector

W, H = config.PROJ_W, config.PROJ_H
CX, CY = config.TARGET_CENTER


def _proj(game, now, calibrating=False):
    return render_projector(W, H, config.TARGET_CENTER, config.RADII, game.total_score,
                            game.hits, game.cooldown_progress(now), calibrating, now)


def _game_with_history(t0):
    g = GameState()
    for i, (x, y, s) in enumerate([(CX + 180, CY + 90, 10), (CX - 110, CY - 40, 50),
                                   (CX + 20, CY + 150, 50), (CX - 200, CY + 120, 10)]):
        g.register_hit(x, y, s, t0 - 60 + i * 5)
    g.update(t0)
    return g


def dummy_camera():
    """監視画面用のダミーカメラ画像 (壁に台形の投影 + 刺さった弾)。"""
    cam = np.zeros((480, 640, 3), np.uint8)
    cam[:] = np.linspace(60, 25, 480, dtype=np.uint8)[:, None, None]
    quad = np.int32([[105, 75], [535, 68], [548, 412], [96, 404]])
    cv2.fillPoly(cam, [quad], (70, 62, 58))
    cv2.circle(cam, (320, 240), 90, (110, 100, 70), 3, cv2.LINE_AA)
    cv2.ellipse(cam, (350, 220), (5, 16), 0, 0, 360, (200, 80, 20), -1)
    cv2.circle(cam, (350, 238), 5, (0, 160, 255), -1)
    return cam, quad.tolist()


def write_all(outdir):
    os.makedirs(outdir, exist_ok=True)
    t0 = 1_000_000.0
    shots = {}

    g = _game_with_history(t0)
    shots["01_idle"] = _proj(g, t0 + 3.7)

    g.register_hit(CX + 12, CY - 18, 100, t0)
    shots["02_hit100"] = _proj(g, t0 + 0.03)
    shots["03_hit100_0.3s"] = _proj(g, t0 + 0.3)

    g2 = _game_with_history(t0)
    g2.register_hit(CX + 205, CY - 40, 10, t0)
    shots["04_cooldown"] = _proj(g2, t0 + 0.75)
    shots["05_calibrating"] = _proj(g2, t0 + 2.0, calibrating=True)

    cam, corners = dummy_camera()
    from .detector import Settled, blur_gray, find_hit, warp_to_projector
    warped, wg = warp_to_projector(cam, corners)
    bg_cam = cam.copy()
    cv2.rectangle(bg_cam, (330, 195), (365, 250), (70, 62, 58), -1)
    hit, mask = find_hit(blur_gray(warp_to_projector(bg_cam, corners)[0]), warped, frame_gray=wg)
    settled = Settled(hit, "settled", config.SETTLE_FRAMES) if hit is not None else None
    shots["06_monitor"] = render_monitor(cam, corners, 1, -1, warped, mask,
                                         config.CORNER_LABELS, config.CORNER_COLORS,
                                         settled=settled, offset=config.HIT_OFFSET)
    shots["07_monitor_click4"] = render_monitor(cam, corners, 1, 2, warped, mask,
                                                config.CORNER_LABELS, config.CORNER_COLORS,
                                                tracking_frames=2, offset=config.HIT_OFFSET)
    shots["08_monitor_offset"] = render_monitor(cam, corners, 1, -1, warped, mask,
                                                config.CORNER_LABELS, config.CORNER_COLORS,
                                                settled=settled, offset=(0, 4), offset_step=2)

    paths = []
    for name, img in shots.items():
        p = os.path.abspath(os.path.join(outdir, name + ".png"))
        cv2.imwrite(p, img)
        paths.append(p)
    return paths


def bench(frames=300):
    """投影画面1フレームの平均描画時間 (ms)。待機中と命中演出中を交互に。"""
    t0 = 2_000_000.0
    g = _game_with_history(t0)
    _proj(g, t0)  # キャッシュを温める
    times = {"idle": [], "hit": []}
    for i in range(frames):
        now = t0 + i / 60
        if i % 90 == 0:
            g.register_hit(CX + (i % 7) * 10, CY - 20, 100 if i % 180 == 0 else 50, now)
        g.update(now)
        s = time.perf_counter()
        _proj(g, now)
        dt = (time.perf_counter() - s) * 1000
        times["idle" if g.ready else "hit"].append(dt)
    allt = times["idle"] + times["hit"]
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in [("all", allt)] + list(times.items())}, max(allt)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    outdir = argv[0] if argv else "snapshots"
    for p in write_all(outdir):
        print(p)
    avg, worst = bench()
    print(f"render_projector 平均 {avg['all']:.2f} ms (待機 {avg['idle']:.2f} / 演出中 {avg['hit']:.2f}), "
          f"最大 {worst:.2f} ms")


if __name__ == "__main__":
    main()
