import cv2
import numpy as np
import time
import sys
import os
import json
import platform
import traceback

# Windows用ビープ音
if platform.system() == "Windows":
    try:
        import winsound
    except ImportError:
        winsound = None
else:
    winsound = None

# --- 設定値 ---
CAMERA_ID = 0

# プロジェクター画面の解像度（全画面対応）
PROJ_W = 1280
PROJ_H = 720

# プロジェクター画面の四つ角基準座標 (0:左上, 1:右上, 2:右下, 3:左下)
PROJ_CORNERS = np.float32([
    [0, 0],              # 0: 左上
    [PROJ_W, 0],         # 1: 右上
    [PROJ_W, PROJ_H],    # 2: 右下
    [0, PROJ_H]          # 3: 左下
])

CORNER_COLORS = [
    (0, 0, 255),      # 0: 赤 (左上)
    (0, 255, 0),      # 1: 緑 (右上)
    (255, 140, 0),    # 2: 青/シアン (右下)
    (0, 255, 255)     # 3: 黄 (左下)
]
CORNER_LABELS = ["[1] TOP-LEFT", "[2] TOP-RIGHT", "[3] BOTTOM-RIGHT", "[4] BOTTOM-LEFT"]

CONFIG_FILE = "corners.json"

# カメラ画面上の対応する4点 (初期値)
camera_corners = [
    [100, 70],   # 左上
    [540, 70],   # 右上
    [540, 410],  # 右下
    [100, 410]   # 左下
]
selected_corner_idx = 0

# キャリブレーションモード
CALIB_IDLE = 0
CALIB_STEP_DARK = 1    # 通常（暗）フレーム撮影
CALIB_STEP_FLASH = 2   # 白点灯フレーム撮影
CALIB_STEP_DONE = 3
calib_state = CALIB_IDLE
calib_start_time = 0
frame_dark_gray = None

# クリックで4点を置くモード
click_step = -1

# --- ナーフ弾（青色ボディ ＋ オレンジ色先端）のHSV色範囲 ---
LOWER_BLUE = np.array([95, 70, 60])
UPPER_BLUE = np.array([135, 255, 255])
LOWER_ORANGE = np.array([5, 90, 80])
UPPER_ORANGE = np.array([25, 255, 255])

# ゲーム状態
STATE_READY = "READY"
STATE_COOLDOWN = "COOLDOWN"
current_state = STATE_READY
cooldown_end_time = 0

total_score = 0
score_history = []
effects = []
bg_warped_gray = None

def load_corners():
    global camera_corners
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if len(data) == 4:
                    camera_corners = data
                    print(f"★ 保存された角の設定を読み込みました: {camera_corners}")
        except Exception:
            pass

def save_corners():
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(camera_corners, f)
    except Exception:
        pass

def order_points(pts):
    """4つの点を厳密に (0:左上, 1:右上, 2:右下, 3:左下) に幾何学ソートする"""
    pts = np.array(pts, dtype="float32")
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = pts[:, 0] - pts[:, 1]
    rect[1] = pts[np.argmax(diff)]
    rect[3] = pts[np.argmin(diff)]
    return rect.tolist()

def play_sound(score):
    if winsound:
        try:
            if score == 100:
                winsound.Beep(1200, 150)
            elif score == 50:
                winsound.Beep(880, 150)
            elif score == 10:
                winsound.Beep(587, 150)
            elif score == -1: # キャリブレーション完了音
                winsound.Beep(1000, 100)
                winsound.Beep(1500, 150)
        except Exception:
            pass

def start_auto_calibration():
    """フラッシュ自動キャリブレーションを開始する"""
    global calib_state, calib_start_time, frame_dark_gray, click_step
    click_step = -1
    calib_state = CALIB_STEP_DARK
    calib_start_time = time.time()
    frame_dark_gray = None
    print("=========================================================")
    print("★ 四隅の自動認識キャリブレーションを開始します...")
    print("  プロジェクター画面が一瞬光り、的の四つ角を完全自動で認識・一致させます！")
    print("=========================================================")

def reset_game():
    global total_score, score_history, effects, bg_warped_gray, current_state
    total_score = 0
    score_history.clear()
    effects.clear()
    bg_warped_gray = None
    current_state = STATE_READY
    print("★ スコアと背景をリセットしました！")

# カメラ画面のマウス操作
def on_camera_mouse(event, x, y, flags, param):
    global camera_corners, selected_corner_idx, click_step

    if event == cv2.EVENT_RBUTTONDOWN:
        reset_game()
        return

    # [RESET] ボタン (x:10~80, y:10~38)
    if event == cv2.EVENT_LBUTTONDOWN and (10 <= x <= 80 and 10 <= y <= 38):
        reset_game()
        return

    # [AUTO ALIGN] ボタン (x:90~220, y:10~38)
    if event == cv2.EVENT_LBUTTONDOWN and (90 <= x <= 220 and 10 <= y <= 38):
        start_auto_calibration()
        return

    # [CLICK 4-PT] ボタン (x:230~350, y:10~38)
    if event == cv2.EVENT_LBUTTONDOWN and (230 <= x <= 350 and 10 <= y <= 38):
        click_step = 0
        print("★ 4点クリック設定モード開始: カメラ画面の [1]左上 をクリックしてください")
        return

    # 4点順番クリックモード
    if click_step >= 0 and event == cv2.EVENT_LBUTTONDOWN:
        camera_corners[click_step] = [x, y]
        print(f"★ {CORNER_LABELS[click_step]} を設定: ({x}, {y})")
        click_step += 1
        if click_step >= 4:
            click_step = -1
            save_corners()
            reset_game()
            play_sound(-1)
            print("★ 4つの角の手動設定が完了しました！")
        return

    # 通常のマウスクリック操作: 最も近い角を選択してその位置に動かす
    if event == cv2.EVENT_LBUTTONDOWN:
        min_dist = 9999
        closest_idx = 0
        for i, pt in enumerate(camera_corners):
            dist = np.hypot(x - pt[0], y - pt[1])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        selected_corner_idx = closest_idx
        camera_corners[selected_corner_idx] = [x, y]
        save_corners()

    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        camera_corners[selected_corner_idx] = [x, y]
        save_corners()

    elif event == cv2.EVENT_LBUTTONUP:
        save_corners()
        reset_game()

def on_proj_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_RBUTTONDOWN:
        reset_game()

def init_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if platform.system() == "Windows" else [cv2.CAP_ANY]
    for cam_id in [CAMERA_ID, 1, 2, 0]:
        for backend in backends:
            cap = cv2.VideoCapture(cam_id, backend)
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    print(f"カメラ (ID: {cam_id}) に接続しました。")
                    return cap
                cap.release()
    return None

def draw_corner_bracket(img, pt, color, label, size=30, thick=3):
    """角をわかりやすく示すL字ブラケットマーカー"""
    x, y = int(pt[0]), int(pt[1])
    # 円と十字
    cv2.circle(img, (x, y), 8, color, -1)
    cv2.circle(img, (x, y), size, (255, 255, 255), thick)
    cv2.circle(img, (x, y), size, color, thick - 1)
    cv2.line(img, (x - size - 8, y), (x + size + 8, y), color, 2)
    cv2.line(img, (x, y - size - 8), (x, y + size + 8), color, 2)
    cv2.putText(img, label, (x + 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1)

def main():
    global camera_corners, selected_corner_idx, total_score, score_history, effects, bg_warped_gray
    global current_state, cooldown_end_time, calib_state, calib_start_time, frame_dark_gray, click_step

    load_corners()

    cap = init_camera()
    if cap is None:
        print("エラー: カメラが見つかりませんでした。")
        input("Enterキーを押すと終了します...")
        return

    win_cam = "Camera Monitor (Control)"
    win_proj = "Projector View (Target)"

    cv2.namedWindow(win_cam, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(win_proj, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_proj, PROJ_W, PROJ_H)

    cv2.setMouseCallback(win_cam, on_camera_mouse)
    cv2.setMouseCallback(win_proj, on_proj_mouse)

    print("=========================================================")
    print(" ナーフ的当てゲーム: 的の四つ角自動認識システム 起動     ")
    print("=========================================================")
    print(" ★ [C] キー / [AUTO ALIGN]: プロジェクターの的を一瞬光らせて四隅を完全自動一致！")
    print(" ★ [M] キー / [CLICK 4-PT]: カメラ画面上で4回クリックして手動一発一致！")
    print(" ★ [R] キー / 右クリック  : スコア＆背景リセット")
    print(" ★ [F] キー               : プロジェクター画面の全画面表示切替")
    print(" ★ [ESC / Q]              : 終了")
    print("=========================================================")

    target_center = (PROJ_W // 2, PROJ_H // 2)
    r100 = 60
    r50 = 150
    r10 = 260

    is_fullscreen = False

    while True:
        try:
            prop_cam = cv2.getWindowProperty(win_cam, cv2.WND_PROP_AUTOSIZE)
            prop_proj = cv2.getWindowProperty(win_proj, cv2.WND_PROP_AUTOSIZE)
            if prop_cam < 0 or prop_proj < 0:
                break
        except Exception:
            pass

        ret, cam_frame = cap.read()
        if not ret:
            break

        cam_gray = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2GRAY)
        current_time = time.time()

        # =========================================================
        # ★ 四つ角の自動認識シーケンス（フラッシュ・キャリブレーション）
        # =========================================================
        if calib_state == CALIB_STEP_DARK:
            # 準備（暗フレーム保存）
            frame_dark_gray = cam_gray.copy()
            calib_state = CALIB_STEP_FLASH
            calib_start_time = current_time

        elif calib_state == CALIB_STEP_FLASH:
            # プロジェクターを白く点灯させて0.25秒待機（カメラ露出安定待ち）
            if current_time - calib_start_time > 0.25:
                frame_bright_gray = cam_gray.copy()
                # 差分を計算（プロジェクターの光った四角形領域を完璧に抽出！）
                diff = cv2.absdiff(frame_bright_gray, frame_dark_gray)
                _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)
                thresh = cv2.dilate(thresh, None, iterations=3)

                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    # 最大面積の輪郭（プロジェクターのスクリーン領域そのもの）
                    best_c = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(best_c)
                    if area > 8000:
                        peri = cv2.arcLength(best_c, True)
                        approx = cv2.approxPolyDP(best_c, 0.03 * peri, True)

                        detected_quad = None
                        if len(approx) == 4:
                            detected_quad = approx.reshape(4, 2)
                        else:
                            # 4頂点近似が難しい場合は最小外接矩形
                            rect = cv2.minAreaRect(best_c)
                            detected_quad = cv2.boxPoints(rect)

                        if detected_quad is not None:
                            ordered = order_points(detected_quad)
                            camera_corners = [[int(p[0]), int(p[1])] for p in ordered]
                            save_corners()
                            play_sound(-1)
                            print(f"★ 成功: プロジェクターの四つ角を完全に認識して一致させました！\n{camera_corners}")

                calib_state = CALIB_IDLE
                reset_game()

        # --- 1. 透視変換（Camera -> Projector）の計算 ---
        warped_bgr = None
        warped_gray = None
        if len(camera_corners) == 4:
            try:
                src_pts = np.float32(camera_corners)
                dst_pts = PROJ_CORNERS
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                warped_bgr = cv2.warpPerspective(cam_frame, M, (PROJ_W, PROJ_H))
                warped_gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
                warped_gray = cv2.GaussianBlur(warped_gray, (21, 21), 0)

                if bg_warped_gray is None:
                    bg_warped_gray = warped_gray.copy()
            except Exception:
                warped_bgr = None
                warped_gray = None

        # --- 2. ナーフ弾（青＋オレンジ）着弾検知 ---
        dart_detected_mask = None
        if calib_state == CALIB_IDLE:
            if current_state == STATE_COOLDOWN:
                if current_time >= cooldown_end_time:
                    if warped_gray is not None:
                        bg_warped_gray = warped_gray.copy()
                    current_state = STATE_READY

            elif current_state == STATE_READY and warped_bgr is not None and bg_warped_gray is not None:
                try:
                    diff = cv2.absdiff(bg_warped_gray, warped_gray)
                    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    thresh = cv2.dilate(thresh, None, iterations=2)

                    # ナーフ弾色検出（青色＋オレンジ色）
                    hsv_warped = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2HSV)
                    mask_blue = cv2.inRange(hsv_warped, LOWER_BLUE, UPPER_BLUE)
                    mask_orange = cv2.inRange(hsv_warped, LOWER_ORANGE, UPPER_ORANGE)
                    mask_dart_color = cv2.bitwise_or(mask_blue, mask_orange)

                    # 変化 × 弾の色（影や人の動きを排除）
                    hit_candidate_mask = cv2.bitwise_and(thresh, mask_dart_color)
                    hit_candidate_mask = cv2.dilate(hit_candidate_mask, None, iterations=2)
                    dart_detected_mask = hit_candidate_mask

                    target_mask = np.zeros((PROJ_H, PROJ_W), dtype=np.uint8)
                    cv2.circle(target_mask, target_center, r10 + 15, 255, -1)
                    final_mask = cv2.bitwise_and(hit_candidate_mask, hit_candidate_mask, mask=target_mask)

                    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for c in contours:
                        area = cv2.contourArea(c)
                        if 40 < area < 8000:
                            (x, y), _ = cv2.minEnclosingCircle(c)
                            hit_x, hit_y = int(x), int(y)

                            distance = np.hypot(hit_x - target_center[0], hit_y - target_center[1])

                            if distance <= r10:
                                if distance <= r100:
                                    score = 100
                                    msg = "100 PTS! (BULLSEYE!)"
                                elif distance <= r50:
                                    score = 50
                                    msg = "50 PTS!"
                                else:
                                    score = 10
                                    msg = "10 PTS!"

                                total_score += score
                                score_history.append((hit_x, hit_y, score))
                                effects.append({'x': hit_x, 'y': hit_y, 'score': score, 'msg': msg, 'start_time': current_time})
                                print(f"★ ナーフ弾命中! 座標:({hit_x}, {hit_y}) -> +{score}点 (合計: {total_score})")

                                play_sound(score)

                                current_state = STATE_COOLDOWN
                                cooldown_end_time = current_time + 1.2
                                break
                except Exception as e:
                    print("Hit error:", e)

        # --- 3. プロジェクター画面（Projector View）の描画 ---
        proj_frame = np.zeros((PROJ_H, PROJ_W, 3), dtype=np.uint8)

        if calib_state == CALIB_STEP_FLASH:
            # 自動キャリブレーション中: 全画面を真っ白にしてカメラに四隅を認識させる！
            proj_frame[:] = (255, 255, 255)
            cv2.putText(proj_frame, "AUTO CALIBRATING...", (PROJ_W // 2 - 250, PROJ_H // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 0), 4)
        else:
            proj_frame[:] = (18, 18, 22)

            # 的の描画
            cv2.circle(proj_frame, target_center, r10, (50, 50, 60), -1)
            cv2.circle(proj_frame, target_center, r10, (255, 255, 255), 4)
            cv2.circle(proj_frame, target_center, r50, (30, 80, 100), -1)
            cv2.circle(proj_frame, target_center, r50, (0, 220, 255), 4)
            cv2.circle(proj_frame, target_center, r100, (40, 40, 160), -1)
            cv2.circle(proj_frame, target_center, r100, (0, 0, 255), 5)
            cv2.circle(proj_frame, target_center, 8, (255, 255, 255), -1)

            cv2.putText(proj_frame, "10", (target_center[0] + r50 + 25, target_center[1] + 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(proj_frame, "50", (target_center[0] + r100 + 20, target_center[1] + 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 220, 255), 2)
            cv2.putText(proj_frame, "100", (target_center[0] - 32, target_center[1] + 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2)

            # 着弾痕
            for hx, hy, hs in score_history[-6:]:
                col = (0, 0, 255) if hs == 100 else ((0, 220, 255) if hs == 50 else (255, 255, 255))
                cv2.circle(proj_frame, (hx, hy), 12, col, -1)
                cv2.circle(proj_frame, (hx, hy), 16, (0, 255, 0), 2)
                cv2.putText(proj_frame, f"+{hs}", (hx + 18, hy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 波紋エフェクト
            active_effects = []
            for eff in effects:
                elapsed = current_time - eff['start_time']
                if elapsed < 1.0:
                    active_effects.append(eff)
                    ripple_r = int(20 + elapsed * 90)
                    alpha_col = (0, 0, 255) if eff['score'] == 100 else ((0, 220, 255) if eff['score'] == 50 else (255, 255, 255))
                    cv2.circle(proj_frame, (eff['x'], eff['y']), ripple_r, alpha_col, max(1, int(3 * (1.0 - elapsed))))
                    cv2.putText(proj_frame, eff['msg'], (eff['x'] - 130, eff['y'] - ripple_r - 10),
                                cv2.FONT_HERSHEY_DUPLEX, 1.0, alpha_col, 2)
            effects = active_effects

            # 四隅の枠線と角マーカー
            cv2.rectangle(proj_frame, (4, 4), (PROJ_W - 5, PROJ_H - 5), (0, 255, 0), 3)
            # 四隅のコーナーブラケット
            draw_corner_bracket(proj_frame, (40, 40), CORNER_COLORS[0], "[1] TL", size=30)
            draw_corner_bracket(proj_frame, (PROJ_W - 40, 40), CORNER_COLORS[1], "[2] TR", size=30)
            draw_corner_bracket(proj_frame, (PROJ_W - 40, PROJ_H - 40), CORNER_COLORS[2], "[3] BR", size=30)
            draw_corner_bracket(proj_frame, (40, PROJ_H - 40), CORNER_COLORS[3], "[4] BL", size=30)

            # スコアボード
            cv2.rectangle(proj_frame, (PROJ_W // 2 - 200, 15), (PROJ_W // 2 + 200, 80), (30, 30, 35), -1)
            cv2.rectangle(proj_frame, (PROJ_W // 2 - 200, 15), (PROJ_W // 2 + 200, 80), (0, 255, 0), 2)
            score_text = f"SCORE: {total_score}"
            text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_DUPLEX, 1.3, 3)[0]
            cv2.putText(proj_frame, score_text, (PROJ_W // 2 - text_size[0] // 2, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 0), 3)

            cv2.putText(proj_frame, "[C]: Auto-Align | [Right Click]: Reset | [F]: Fullscreen",
                        (PROJ_W // 2 - 240, PROJ_H - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1)

        # --- 4. カメラ監視・設定画面の描画 ---
        cam_display = cam_frame.copy()

        # UIボタン
        cv2.rectangle(cam_display, (10, 10), (80, 38), (0, 0, 180), -1)
        cv2.putText(cam_display, "RESET", (18, 28), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        cv2.rectangle(cam_display, (90, 10), (220, 38), (0, 150, 0), -1)
        cv2.putText(cam_display, "AUTO ALIGN", (98, 28), cv2.FONT_HERSHEY_DUPLEX, 0.48, (255, 255, 255), 1)

        cv2.rectangle(cam_display, (230, 10), (350, 38), (180, 90, 0), -1)
        cv2.putText(cam_display, "CLICK 4-PT", (240, 28), cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 255, 255), 1)

        # プロジェクター枠線（緑線）
        if len(camera_corners) == 4:
            pts_array = np.array(camera_corners, np.int32).reshape((-1, 1, 2))
            cv2.polylines(cam_display, [pts_array], isClosed=True, color=(0, 255, 0), thickness=2)

        # 四隅ハンドル
        for i, pt in enumerate(camera_corners):
            is_sel = (i == selected_corner_idx)
            draw_corner_bracket(cam_display, pt, CORNER_COLORS[i], CORNER_LABELS[i], size=18, thick=3 if is_sel else 2)

        # 4点クリック中のガイド
        if click_step >= 0:
            cv2.rectangle(cam_display, (10, 410), (630, 475), (0, 0, 0), -1)
            cv2.putText(cam_display, f">> CLICK {CORNER_LABELS[click_step]} ON CAMERA VIEW <<", (20, 445),
                        cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 255), 2)
        else:
            curr_label = CORNER_LABELS[selected_corner_idx]
            cv2.putText(cam_display, f"[C] or 'AUTO ALIGN': Flash & Auto-match 4 corners", (10, 445),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(cam_display, f"Click corner to move | Arrow keys: fine-tune ({curr_label})", (10, 468),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

        # 右上に「補正後視界」、右下に「弾センサー」を表示
        if warped_bgr is not None:
            preview = cv2.resize(warped_bgr, (150, 85))
            cv2.rectangle(preview, (0, 0), (150, 85), (0, 255, 0), 1)
            cv2.putText(preview, "WARPED TARGET", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
            cam_display[10:95, 480:630] = preview

        if dart_detected_mask is not None:
            mask_preview = cv2.resize(dart_detected_mask, (150, 85))
            mask_preview_bgr = cv2.cvtColor(mask_preview, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(mask_preview_bgr, (0, 0), (150, 85), (0, 165, 255), 1)
            cv2.putText(mask_preview_bgr, "DART SENSOR", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
            cam_display[105:190, 480:630] = mask_preview_bgr

        cv2.imshow(win_cam, cam_display)
        cv2.imshow(win_proj, proj_frame)

        # キー操作
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q') or key == ord('Q'):
            break
        elif key == ord('c') or key == ord('C'):
            start_auto_calibration()
        elif key == ord('m') or key == ord('M'):
            click_step = 0
            print("★ 4点クリック設定モード開始: カメラ画面の [1]左上 をクリックしてください")
        elif key == ord('r') or key == ord('R'):
            reset_game()
        elif key == 9:  # TABキー
            selected_corner_idx = (selected_corner_idx + 1) % 4
            print(f"★ 選択中の角: {CORNER_LABELS[selected_corner_idx]}")
        # 矢印キー微調整
        elif key in [0, 82, 0x260000, 2490368]:  # UP
            camera_corners[selected_corner_idx][1] -= 2
            save_corners()
        elif key in [1, 84, 0x280000, 2621440]:  # DOWN
            camera_corners[selected_corner_idx][1] += 2
            save_corners()
        elif key in [2, 81, 0x250000, 2424832]:  # LEFT
            camera_corners[selected_corner_idx][0] -= 2
            save_corners()
        elif key in [3, 83, 0x270000, 2555904]:  # RIGHT
            camera_corners[selected_corner_idx][0] += 2
            save_corners()
        elif key == ord('w') or key == ord('W'):
            camera_corners[selected_corner_idx][1] -= 2
            save_corners()
        elif key == ord('s') or key == ord('S'):
            camera_corners[selected_corner_idx][1] += 2
            save_corners()
        elif key == ord('d') or key == ord('D'):
            camera_corners[selected_corner_idx][0] += 2
            save_corners()
        elif key == ord('f') or key == ord('F'):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(win_proj, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(win_proj, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("エラー:")
        traceback.print_exc()
        input("Enterキーを押すと終了します...")
