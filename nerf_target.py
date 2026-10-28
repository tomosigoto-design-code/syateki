"""ナーフ射的ゲームの起動スクリプト。

    python nerf_target.py          # 本番 (カメラ + プロジェクター)
    python nerf_target.py --demo   # カメラなし。投影画面のクリックで着弾
"""
import traceback

from syateki.app import main

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("エラー:")
        traceback.print_exc()
        input("Enterキーを押すと終了します...")
