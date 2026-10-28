"""会場で変える設定 (settings.json) の読み書き。今は HIT_OFFSET だけ。"""
import json
import os

from . import config


def load_offset(path=config.SETTINGS_FILE):
    """settings.json の hit_offset を (dx, dy) で返す。無い・壊れているときは config.HIT_OFFSET。"""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                dx, dy = json.load(f)["hit_offset"]
            offset = (int(dx), int(dy))
            print(f"★ 着弾補正を読み込みました: {offset}")
            return offset
        except Exception:
            pass
    return tuple(int(v) for v in config.HIT_OFFSET)


def save_offset(offset, path=config.SETTINGS_FILE):
    """hit_offset を保存する。他のキーがあれば残す。"""
    data = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            data = {}
    except Exception:
        pass
    data["hit_offset"] = [int(offset[0]), int(offset[1])]
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass
