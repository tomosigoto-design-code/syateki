# ナーフ射的ゲーム (Nerf Target Projection Mapping Game)

壁に投影した的をナーフで撃つと、カメラが着弾を見つけて得点・演出を出すゲーム。

- 投影画面 (`Projector View`): ネオン調の的、スコアHUD、命中演出。`F` で全画面。
- 監視画面 (`Camera Monitor`): カメラ映像、四隅の調整、補正後映像 (WARPED)、弾センサー (DART SENSOR)。
- 四隅合わせ: プロジェクターを一瞬白く光らせて自動認識 (フラッシュ式)。手動の4点クリック、キーでの微調整もできる。
- 検出: 背景との差分 × 弾の色 (青ボディ / オレンジ先端) で候補を出す。着弾点はオレンジ先端の重心 (無ければ候補の最下点) で、`config.HIT_OFFSET` で補正できる。

## 起動

Windows: `run_game.bat` をダブルクリック (`requirements.txt` を入れてから起動)。

```bash
pip install -r requirements.txt
python nerf_target.py          # 本番
python nerf_target.py --demo   # カメラなし。投影画面をクリックすると着弾 (リハーサル用)
```

## 操作

| 操作 | 機能 |
| :--- | :--- |
| `C` / `AUTO ALIGN` | 四隅の自動認識 (投影が一瞬白くなる) |
| `M` / `CLICK 4-PT` | 監視画面で 左上→右上→右下→左下 の順にクリックして設定 |
| クリック / ドラッグ | 監視画面で一番近い角を動かす (離したときに保存) |
| `TAB` | 微調整する角を切り替え |
| 矢印キー / `WASD` | 選択中の角を2pxずつ動かす |
| `R` / 右クリック / `RESET` | スコアと背景をリセット |
| `F` | 投影画面の全画面切り替え |
| `ESC` / `Q` | 終了 |

四隅の設定は `corners.json` に保存され、次回起動時に読み込まれる。

## 構成

```
nerf_target.py          起動スクリプト
syateki/
  config.py             定数 (検出のしきい値・的の半径など)
  calibration.py        四隅の並べ替え・フラッシュ検出・corners.json
  detector.py           着弾検出・スコア判定
  game.py               スコア・履歴・READY/COOLDOWN
  render_projector.py   投影画面の描画
  render_monitor.py     監視画面の描画
  sound.py              効果音
  keys.py               キーコード判定
  app.py                メインループ・入力
  snapshot.py           画面のPNG出力と描画時間の計測
tests/                  pytest
```

```bash
pip install pytest
python -m pytest                       # テスト
python -m syateki.snapshot snapshots   # 画面をPNGで書き出し、描画時間を表示
```

## 必要なもの

- Windows 10 / 11 (Linux でも動く)、Python 3.8 以上
- `opencv-python` 5.0 以上 (4.x でも動くが、文字の描画が粗くなる)、`numpy`
- Webカメラ、プロジェクター、ナーフ (エリートダーツ)
