"""メインループと入力処理。"""
import argparse
import platform
import time

import cv2
import numpy as np

from . import config
from .calibration import detect_projection_corners, load_corners, save_corners
from .detector import HitTracker, find_hit, score_for_point, warp_to_projector
from .game import COOLDOWN, GameState
from .keys import decode_key
from .render_monitor import BUTTONS, render_monitor
from .render_projector import render_projector
from .settings import MODES, load_mode, load_offset, save_mode, save_offset
from . import sound

WIN_CAM = "Camera Monitor (Control)"
WIN_PROJ = "Projector View (Target)"

CALIB_IDLE, CALIB_DARK, CALIB_FLASH = 0, 1, 2
# O キーのワンクリック補正: 待機 → 着弾待ち → 投影画面のクリック待ち
OFS_IDLE, OFS_WAIT_SHOT, OFS_WAIT_CLICK = 0, 1, 2

_NUDGE = {"up": (0, -1), "w": (0, -1), "down": (0, 1), "s": (0, 1),
          "left": (-1, 0), "a": (-1, 0), "right": (1, 0), "d": (1, 0)}
_OFFSET_KEYS = {"i": (0, -1), "k": (0, 1), "j": (-1, 0), "l": (1, 0)}


def _bgm(name):
    """sound.bgm_start / bgm_stop を呼ぶ。無い版の sound なら何もしない。"""
    fn = getattr(sound, name, None)
    if callable(fn):
        try:
            fn()
        except Exception:
            pass


def _in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


class App:
    """状態をすべて持つ。step() はウィンドウを使わないのでテストから呼べる。"""

    def __init__(self, demo=False, corners_path=config.CONFIG_FILE,
                 settings_path=config.SETTINGS_FILE):
        self.demo = demo
        self.corners_path = corners_path
        self.corners = load_corners(corners_path)
        self.settings_path = settings_path
        self.hit_offset = load_offset(settings_path)
        self.mode = load_mode(settings_path)  # 演出モード "normal" / "fever"
        self.tracker = HitTracker()
        self.last_settled = None  # 直近の確定結果 (監視画面の表示用)
        self.offset_step = OFS_IDLE
        self.offset_ref = None    # ワンクリック補正: 検出した補正前の着弾点
        self.selected = 0
        self.click_step = -1      # 0〜3: 4点クリック設定中
        self.dragging = False
        self.calib_step = CALIB_IDLE
        self.calib_start = 0.0
        self.dark_gray = None
        self.bg_gray = None
        self.game = GameState()
        self.fullscreen = False
        self.running = True

    # ------------------------------------------------------------ 操作
    def reset_game(self, quiet=False):
        """スコアと背景をリセット。quiet=True は角の設定変更に伴うリセット (音なし)。"""
        had_bonus = self.game.bonus_until is not None
        self.game.reset()
        if had_bonus:
            _bgm("bgm_stop")
        self.bg_gray = None
        self.tracker.reset()
        if not quiet:
            sound.play("reset")
        print("★ スコアと背景をリセットしました！")

    def start_auto_calibration(self, now=None):
        self.click_step = -1
        self.calib_step = CALIB_DARK
        self.calib_start = time.time() if now is None else now
        self.dark_gray = None
        sound.play("calib_start")
        print("★ 四隅の自動認識キャリブレーションを開始します (プロジェクターが一瞬光ります)")

    def start_click_mode(self):
        self.click_step = 0
        print("★ 4点クリック設定モード開始: カメラ画面の [1]左上 をクリックしてください")

    def save(self):
        save_corners(self.corners, self.corners_path)

    def set_offset(self, dx, dy):
        self.hit_offset = (int(dx), int(dy))
        save_offset(self.hit_offset, self.settings_path)
        print(f"★ 着弾補正 HIT_OFFSET = {self.hit_offset}")

    def toggle_mode(self):
        self.mode = MODES[(MODES.index(self.mode) + 1) % len(MODES)]
        save_mode(self.mode, self.settings_path)
        print(f"★ 演出モード: {self.mode}")
        if self.mode != "fever":
            self.game.bull_streak = 0
            if self.game.end_bonus():
                _bgm("bgm_stop")
                sound.play("bonus_end")
                print("★ ボーナスタイム終了 (normal に切替)")

    def toggle_offset_mode(self):
        if self.offset_step == OFS_IDLE:
            self.offset_step = OFS_WAIT_SHOT
            self.offset_ref = None
            print("★ 着弾補正モード: 的を1発撃ち、投影画面で実際に当たった位置をクリック ([O] で中止)")
        else:
            self.offset_step = OFS_IDLE
            print("★ 着弾補正モードを中止しました")

    def _offset_shot(self, raw):
        """補正モード中の着弾。補正前の位置を覚えてクリック待ちへ。"""
        if self.offset_step == OFS_WAIT_SHOT:
            self.offset_ref = raw
            self.offset_step = OFS_WAIT_CLICK
            print(f"★ 検出位置 (補正前) ({raw[0]:.0f}, {raw[1]:.0f})。投影画面で実際の着弾位置をクリック")

    def on_camera_mouse(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.reset_game()
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if _in_rect(x, y, BUTTONS["reset"]):
                self.reset_game()
                return
            if _in_rect(x, y, BUTTONS["auto"]):
                self.start_auto_calibration()
                return
            if _in_rect(x, y, BUTTONS["click4"]):
                self.start_click_mode()
                return
            if self.click_step >= 0:
                self.corners[self.click_step] = [x, y]
                print(f"★ {config.CORNER_LABELS[self.click_step]} を設定: ({x}, {y})")
                self.click_step += 1
                if self.click_step >= 4:
                    self.click_step = -1
                    self.save()
                    self.reset_game(quiet=True)
                    sound.play("calib_ok")
                    print("★ 4つの角の手動設定が完了しました！")
                return
            # 最も近い角をクリック位置へ動かす (保存は離したとき)
            d = [np.hypot(x - p[0], y - p[1]) for p in self.corners]
            self.selected = int(np.argmin(d))
            self.corners[self.selected] = [x, y]
            self.dragging = True
        elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON) and self.dragging:
            self.corners[self.selected] = [x, y]
        elif event == cv2.EVENT_LBUTTONUP and self.dragging:
            self.dragging = False
            self.save()
            self.reset_game(quiet=True)

    def on_proj_mouse(self, event, x, y, flags, param=None, now=None):
        if event == cv2.EVENT_RBUTTONDOWN:
            self.reset_game()
        elif event == cv2.EVENT_LBUTTONDOWN and self.offset_step == OFS_WAIT_CLICK:
            rx, ry = self.offset_ref
            self.offset_step = OFS_IDLE
            self.set_offset(round(x - rx), round(y - ry))
        elif event == cv2.EVENT_LBUTTONDOWN and self.demo:
            # デモ: クリック位置を着弾として扱う (補正はかけない)
            if self.calib_step == CALIB_IDLE and self.game.ready:
                if self._register(x, y, time.time() if now is None else now):
                    self._offset_shot((float(x), float(y)))

    def handle_key(self, code):
        """キー1つを処理する。終了なら False。"""
        k = decode_key(code)
        if k is None:
            return True
        if k in ("esc", "q"):
            return False
        if k == "c":
            self.start_auto_calibration()
        elif k == "m":
            self.start_click_mode()
        elif k == "r":
            self.reset_game()
        elif k == "tab":
            self.selected = (self.selected + 1) % 4
            print(f"★ 選択中の角: {config.CORNER_LABELS[self.selected]}")
        elif k in _NUDGE:
            dx, dy = _NUDGE[k]
            p = self.corners[self.selected]
            p[0] += dx * config.NUDGE_PX
            p[1] += dy * config.NUDGE_PX
            self.save()
            self.bg_gray = None  # 変換が変わるので背景を撮り直す
            self.tracker.reset()
        elif k in _OFFSET_KEYS:
            dx, dy = _OFFSET_KEYS[k]
            self.set_offset(self.hit_offset[0] + dx * config.OFFSET_STEP_PX,
                            self.hit_offset[1] + dy * config.OFFSET_STEP_PX)
        elif k == "o":
            self.toggle_offset_mode()
        elif k == "p":
            self.toggle_mode()
        elif k == "f":
            self.toggle_fullscreen()
        return True

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        try:
            cv2.setWindowProperty(WIN_PROJ, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        except cv2.error:
            pass

    # ------------------------------------------------------------ 1フレーム
    def _register(self, x, y, now):
        score = score_for_point(x, y)
        if score <= 0:
            return 0
        mult, event = self.game.register_hit(x, y, score, now, fever=self.mode == "fever")
        bonus = f" x{mult} (ボーナス残り {self.game.bonus_left(now):.1f}秒)" if mult > 1 else ""
        print(f"★ ナーフ弾命中! 座標:({x}, {y}) -> +{score * mult}点{bonus} (合計: {self.game.total_score})")
        sound.play("hit100_fever" if score == 100 and self.mode == "fever" else f"hit{score}")
        if event == "start":
            sound.play("bonus_start")
            _bgm("bgm_start")
            print(f"★ ボーナスタイム開始! {config.BONUS_SEC:.0f}秒間スコア2倍")
        elif event == "extend":  # BGM は止めずに流し続ける
            print(f"★ ボーナスタイム延長! 残り {self.game.bonus_left(now):.1f}秒")
        return score

    def _on_settled(self, settled, now):
        self.last_settled = settled
        h = settled.hit
        if h is None:
            print(f"★ 弾を見失ったので無効 (lost, {settled.frames}f, SETTLE_LOST=drop)")
            return
        print(f"★ 着弾確定: {settled.reason} {settled.frames}f / {h.method} 橙{h.orange_px}px / "
              f"補正前({h.raw[0]:.1f}, {h.raw[1]:.1f}) + offset{self.hit_offset} -> ({h.x}, {h.y})")
        if self._register(h.x, h.y, now):
            self._offset_shot(h.raw)

    def _render_projector(self, now):
        args = (config.PROJ_W, config.PROJ_H, config.TARGET_CENTER, config.RADII,
                self.game.total_score, self.game.hits, self.game.cooldown_progress(now),
                self.calib_step == CALIB_FLASH, now)
        return _call_projector(args, mode=self.mode, bonus_start=self.game.bonus_start,
                               bonus_until=self.game.bonus_until)

    def _calibration_step(self, cam_gray, now):
        # 暗フレームは投影を真っ黒にして DARK_WAIT_SEC 待ってから撮る (演出の色が差分を潰さないように)
        if self.calib_step == CALIB_DARK and now - self.calib_start >= config.DARK_WAIT_SEC:
            self.dark_gray = cam_gray.copy()
            self.calib_step = CALIB_FLASH
            self.calib_start = now
        elif self.calib_step == CALIB_FLASH and now - self.calib_start > config.FLASH_WAIT_SEC:
            found = detect_projection_corners(self.dark_gray, cam_gray)
            if found is not None:
                self.corners = found
                self.save()
                sound.play("calib_ok")
                print(f"★ 成功: プロジェクターの四つ角を認識しました {self.corners}")
            else:
                print("★ 四隅を認識できませんでした (設定は変えていません)")
            self.calib_step = CALIB_IDLE
            self.reset_game(quiet=True)

    def step(self, cam_frame, now):
        """カメラ1フレームを処理して (投影画面, 監視画面) を返す。"""
        self._calibration_step(cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY), now)

        warped_bgr = warped_gray = None
        if len(self.corners) == 4:
            try:
                warped_bgr, warped_gray = warp_to_projector(cam_frame, self.corners)
                if self.bg_gray is None:
                    self.bg_gray = warped_gray.copy()
                    self.tracker.reset()
            except Exception:
                warped_bgr = warped_gray = None

        if self.game.update_bonus(now):
            _bgm("bgm_stop")
            sound.play("bonus_end")
            print("★ ボーナスタイム終了")

        dart_mask = None
        if self.calib_step == CALIB_IDLE:
            if self.game.phase == COOLDOWN:
                if self.game.update(now) and warped_gray is not None:
                    self.bg_gray = warped_gray.copy()
            elif warped_bgr is not None and self.bg_gray is not None:
                try:
                    hit, dart_mask = find_hit(self.bg_gray, warped_bgr, frame_gray=warped_gray,
                                              offset=self.hit_offset)
                    # 止まるまで追跡し、確定するまで COOLDOWN にも背景更新にも入らない
                    settled = self.tracker.update(hit, now)
                    if settled is not None:
                        self._on_settled(settled, now)
                except Exception as e:
                    print("Hit error:", e)

        if self.calib_step == CALIB_DARK:
            proj = np.zeros((config.PROJ_H, config.PROJ_W, 3), np.uint8)
        else:
            proj = self._render_projector(now)
        monitor = render_monitor(cam_frame, self.corners, self.selected, self.click_step,
                                 warped_bgr, dart_mask, config.CORNER_LABELS, config.CORNER_COLORS,
                                 settled=self.last_settled,
                                 tracking_frames=self.tracker.seen,
                                 offset=self.hit_offset, offset_step=self.offset_step,
                                 mode=self.mode, bonus_left=self.game.bonus_left(now))
        return proj, monitor


def _call_projector(args, **kw):
    """新しい引数を受け取れない版の render_projector でも落ちないよう、後ろから外して呼び直す。"""
    keys = list(kw)
    while True:
        try:
            return render_projector(*args, **{k: kw[k] for k in keys})
        except TypeError:
            if not keys:
                raise
            keys.pop()


# ---------------------------------------------------------------- 起動
def init_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
    for cam_id in [config.CAMERA_ID, 1, 2, 0]:
        for backend in backends:
            cap = cv2.VideoCapture(cam_id, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_SIZE[0])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_SIZE[1])
                    print(f"カメラ (ID: {cam_id}) に接続しました。")
                    return cap
                cap.release()
    return None


def demo_frame():
    """--demo 用のダミーカメラ画像。"""
    img = np.full((config.CAMERA_SIZE[1], config.CAMERA_SIZE[0], 3), 40, np.uint8)
    cv2.putText(img, "DEMO MODE - NO CAMERA", (120, 250), cv2.FONT_HERSHEY_DUPLEX, 0.8,
                (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(img, "click the projector window to shoot", (130, 285), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (160, 160, 160), 1, cv2.LINE_AA)
    return img


def _windows_closed():
    try:
        return (cv2.getWindowProperty(WIN_CAM, cv2.WND_PROP_AUTOSIZE) < 0
                or cv2.getWindowProperty(WIN_PROJ, cv2.WND_PROP_AUTOSIZE) < 0)
    except cv2.error:
        return False


def print_help(demo):
    print("=========================================================")
    print(" ナーフ射的ゲーム" + (" [DEMO: 投影画面クリックで着弾]" if demo else ""))
    print("=========================================================")
    print(" [C] / AUTO ALIGN : プロジェクターを光らせて四隅を自動認識")
    print(" [M] / CLICK 4-PT : カメラ画面で4点クリックして手動設定")
    print(" クリック/ドラッグ: 一番近い角を動かす")
    print(" [TAB]            : 微調整する角を切替")
    print(" 矢印 / WASD      : 選択中の角を2pxずつ移動")
    print(" [R] / 右クリック : スコアと背景をリセット")
    print(" [F]              : 投影画面の全画面切替")
    print(" [I][K] / [J][L]  : 着弾補正を上下 / 左右に2pxずつ (settings.json に保存)")
    print(" [O]              : 着弾補正モード (1発撃って投影画面で実際の位置をクリック)")
    print(" [P]              : 演出モード切替 (normal / fever)")
    print(" [ESC] / [Q]      : 終了")
    print("=========================================================")


def run(demo=False):
    sound.init()
    app = App(demo=demo)
    cap = None
    if not demo:
        cap = init_camera()
        if cap is None:
            print("エラー: カメラが見つかりませんでした。 (カメラなしで試すなら --demo)")
            input("Enterキーを押すと終了します...")
            return

    cv2.namedWindow(WIN_CAM, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(WIN_PROJ, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_PROJ, config.PROJ_W, config.PROJ_H)
    cv2.setMouseCallback(WIN_CAM, app.on_camera_mouse)
    cv2.setMouseCallback(WIN_PROJ, app.on_proj_mouse)
    print_help(demo)

    dummy = demo_frame() if demo else None
    try:
        while True:
            if _windows_closed():
                break
            if demo:
                frame = dummy
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            proj, monitor = app.step(frame, time.time())
            cv2.imshow(WIN_CAM, monitor)
            cv2.imshow(WIN_PROJ, proj)
            if not app.handle_key(cv2.waitKeyEx(1)):
                break
    finally:
        _bgm("bgm_stop")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def main(argv=None):
    p = argparse.ArgumentParser(description="ナーフ射的 プロジェクションマッピングゲーム")
    p.add_argument("--demo", action="store_true", help="カメラなし。投影画面のクリックを着弾として扱う")
    args = p.parse_args(argv)
    run(demo=args.demo)
