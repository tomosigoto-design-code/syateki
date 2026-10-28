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
# 橙が足りないときの着弾点: 輪郭の重心と最下点の混ぜ具合 (0=重心, 1=最下点)
FALLBACK_BLEND = 0.5
# 着弾点に足す補正 (投影座標px, (dx, dy))。会場でカメラ位置のせいで当たりが
# ずれるときに使う。例: 実際より上に出るなら (0, 10)。
# settings.json があればその値を使う (会場で I/J/K/L・O キーで調整して保存)。
HIT_OFFSET = (0, 0)
# 着弾の安定待ち: 候補が出ても即確定せず、止まるまで追跡する
SETTLE_FRAMES = 3     # 連続このフレーム数、
SETTLE_PX = 6         # 着弾点がこの距離 (投影座標px) 以内に収まったら確定
SETTLE_GRACE = 3      # 追跡中、この連続フレーム数見えなければ「見失った」(飛行中のちらつき対策)
SETTLE_LOST = "last"  # 途中で消えたら "last": 最後に見えた位置で確定 / "drop": 無効
SETTLE_TIMEOUT = 0.5  # 追跡がこの秒数を超えたら、その時点の位置で確定
SETTINGS_FILE = "settings.json"  # corners.json と同じくカレントディレクトリ
OFFSET_STEP_PX = 2  # I/J/K/L で HIT_OFFSET を動かす量

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
