"""会場で変える設定 (settings.json) の読み書き。

{"hit_offset": [dx, dy], "mode": "normal" | "fever"}。片方だけ保存しても他のキーは残す。
"""
import json
import os

from . import config

MODES = ("normal", "fever")


def _read(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _update(path, **values):
    data = _read(path)
    data.update(values)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except Exception:
        pass


def load_offset(path=config.SETTINGS_FILE):
    """hit_offset を (dx, dy) で返す。無い・壊れているときは config.HIT_OFFSET。"""
    try:
        dx, dy = _read(path)["hit_offset"]
        offset = (int(dx), int(dy))
        print(f"★ 着弾補正を読み込みました: {offset}")
        return offset
    except Exception:
        return tuple(int(v) for v in config.HIT_OFFSET)


def save_offset(offset, path=config.SETTINGS_FILE):
    _update(path, hit_offset=[int(offset[0]), int(offset[1])])


def load_mode(path=config.SETTINGS_FILE):
    """演出モード "normal" / "fever"。無い・不正なら "normal"。"""
    mode = _read(path).get("mode")
    return mode if mode in MODES else MODES[0]


def save_mode(mode, path=config.SETTINGS_FILE):
    _update(path, mode=mode)
