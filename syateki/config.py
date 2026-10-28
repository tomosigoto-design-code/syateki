"""定数。検出系の数値は実機で調整済みなので変えないこと。"""
import numpy as np

CAMERA_ID = 0
CAMERA_SIZE = (640, 480)  # 要求するカメラ解像度 (実際の値は機種次第)

# プロジェクター画面の解像度
PROJ_W = 1280
PROJ_H = 720

# 投影画面の四隅 (0:左上, 1:右上, 2:右下, 3:左下)
PROJ_CORNERS = np.float32([[0, 0], [PROJ_W, 0], [PROJ_W, PROJ_H], [0, PROJ_H]])

CORNER_COLORS = [
    (0, 0, 255),    # 0: 赤 (左上)
    (0, 255, 0),    # 1: 緑 (右上)
    (255, 140, 0),  # 2: 青 (右下)
    (0, 255, 255),  # 3: 黄 (左下)
]
CORNER_LABELS = ["[1] TOP-LEFT", "[2] TOP-RIGHT", "[3] BOTTOM-RIGHT", "[4] BOTTOM-LEFT"]

CONFIG_FILE = "corners.json"  # カレントディレクトリに保存 (run_game.bat はスクリプトの場所へ cd する)

# カメラ画面上の四隅の初期値 (640x480 想定)
DEFAULT_CAMERA_CORNERS = [[100, 70], [540, 70], [540, 410], [100, 410]]

# ナーフ弾 (青ボディ + オレンジ先端) の HSV 範囲
LOWER_BLUE = np.array([95, 70, 60])
UPPER_BLUE = np.array([135, 255, 255])
LOWER_ORANGE = np.array([5, 90, 80])
UPPER_ORANGE = np.array([25, 255, 255])

# 着弾検出
BLUR_KSIZE = (21, 21)
HIT_DIFF_THRESH = 30
HIT_AREA_MIN = 40       # 40 < area < 8000
HIT_AREA_MAX = 8000
TARGET_MASK_MARGIN = 15  # 判定領域 = r10 + 15
ORANGE_MIN_PIXELS = 15   # 橙の画素がこれ以上あれば、その重心を着弾点にする
# 着弾点に足す補正 (投影座標px, (dx, dy))。会場でカメラ位置のせいで当たりが
# ずれるときに使う。例: 実際より上に出るなら (0, 10)。
HIT_OFFSET = (0, 0)

# フラッシュキャリブレーション
CALIB_DIFF_THRESH = 35
CALIB_MIN_AREA = 8000
FLASH_WAIT_SEC = 0.25    # 白点灯後、カメラ露出が安定するまで待つ

COOLDOWN_SEC = 1.2       # 命中後、次の検出までの時間

# 的
TARGET_CENTER = (PROJ_W // 2, PROJ_H // 2)
R100 = 60
R50 = 150
R10 = 260
RADII = (R100, R50, R10)

NUDGE_PX = 2  # 矢印キー / WASD で角を動かす量
