"""効果音。numpy で合成した WAV (FM ベル + 短い残響) をキャッシュし、ゲームループを止めずに鳴らす。

    init()            # 起動時に1回。WAV を書き出す
    play("hit100")    # すぐ返る。失敗しても例外を出さない

名前: hit100 / hit50 / hit10 / calib_ok / calib_start / reset
再生: Windows は winsound.PlaySound (SND_ASYNC)、Linux は aplay / paplay。
どちらも無ければ無音。鳴っている途中の次の音は上書き。
"""
import os
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
import wave

import numpy as np

SR = 44100
PEAK = 10 ** (-3 / 20)  # -3 dBFS
NAMES = ("hit100", "hit50", "hit10", "calib_ok", "calib_start", "reset")

winsound = None
if platform.system() == "Windows":
    try:
        import winsound
    except ImportError:
        winsound = None

_paths = {}
_lock = threading.Lock()
_queue = queue.Queue()
_worker = None
_proc = None
_player = None


# ---- 合成 ----------------------------------------------------------------
# サイン波 + 少しの FM で作るベル/クリスタル音に、短いリバーブ風の尾を足す。

def _t(dur):
    return np.arange(int(SR * dur)) / SR


def _note(n):
    """MIDI ノート番号 → Hz。"""
    return 440.0 * 2 ** ((n - 69) / 12)


def _shape(s, t, attack, decay, release=0.02):
    """立ち上がり (attack 秒) + 指数減衰 + 末尾の半コサインフェード (プツッ防止)。"""
    s = s * np.clip(t / attack, 0, 1) * np.exp(-decay * t)
    n = min(int(SR * release), len(s))
    s[len(s) - n:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n))
    return s


def _fm_bell(freq, dur, ratio=3.5, index=1.0, decay=5.0, attack=0.002, bright_decay=25.0):
    """FM ベル。変調指数は速く減衰し、鳴り始めだけ少しきらっとする。"""
    t = _t(dur)
    idx = index * np.exp(-bright_decay * t)
    s = np.sin(2 * np.pi * freq * t + idx * np.sin(2 * np.pi * freq * ratio * t))
    s += 0.2 * np.sin(2 * np.pi * freq * 2.0 * t) * np.exp(-decay * t)  # 2倍音を少し
    return _shape(s, t, attack, decay)


def _sub(dur, f0=98.0, f1=46.0, decay=5.0):
    """低いサブの「ドゥン」。ピッチを少し落とす。"""
    t = _t(dur)
    f = f1 + (f0 - f1) * np.exp(-18 * t)
    return _shape(np.sin(2 * np.pi * np.cumsum(f) / SR), t, 0.006, decay, release=0.05)


def _lowpass(x, alpha):
    """1次ローパス (alpha 小さいほどこもる)。"""
    y = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += alpha * (v - acc)
        y[i] = acc
    return y


def _reverb(x, tail=0.3, mix=0.3):
    """遅延を数本重ねた減衰コピーで短い残響を付ける。"""
    out = np.concatenate([x, np.zeros(int(SR * tail))])
    wet = np.zeros_like(out)
    for ms, g in [(23, 0.55), (37, 0.45), (53, 0.38), (79, 0.3), (113, 0.22), (157, 0.15), (211, 0.1)]:
        d = int(SR * ms / 1000)
        m = min(len(x), len(out) - d)
        wet[d:d + m] += g * x[:m]
    return out + mix * _lowpass(wet, 0.35)


def _mix(parts):
    """[(開始秒, 信号), ...] を足し合わせる。長さは自動。"""
    n = max(int(SR * st) + len(s) for st, s in parts)
    buf = np.zeros(n)
    for st, s in parts:
        i = int(SR * st)
        buf[i:i + len(s)] += s
    return buf


def _hit10():
    s = _fm_bell(_note(88), 0.2, ratio=4.0, index=0.6, decay=22, bright_decay=45)  # E6 クリスタル
    return _reverb(s, tail=0.12)


def _hit50():
    return _reverb(_mix([
        (0.0, _fm_bell(_note(81), 0.16, ratio=3.5, index=0.7, decay=24, bright_decay=40)),   # A5
        (0.07, _fm_bell(_note(88), 0.2, ratio=3.5, index=0.7, decay=18, bright_decay=40)),   # E6 (完全5度上)
    ]), tail=0.12)


def _hit100():
    # 低い「ドゥン」+ Cmaj9 のアルペジオ (C5 E5 G5 B5 D6)、少し遅れて上の C6 E6
    parts = [(0.0, 0.9 * _sub(0.75, decay=5.0))]
    for k, n in enumerate([72, 76, 79, 83, 86]):
        parts.append((k * 0.055, 0.5 * _fm_bell(_note(n), 0.85 - k * 0.055, index=0.9, decay=4.5)))
    for n in (84, 88):
        parts.append((0.3, 0.25 * _fm_bell(_note(n), 0.55, ratio=7.0, index=0.4, decay=6.0)))
    return _reverb(_mix(parts), tail=0.3, mix=0.35)


def _calib_start():
    """ノイズを共鳴フィルター (状態変数フィルター) でゆっくりスイープ。"""
    dur = 0.7
    n = int(SR * dur)
    t = _t(dur)
    noise = np.random.default_rng(7).standard_normal(n)
    fc = 250 * (2500 / 250) ** (t / dur)
    f = 2 * np.sin(np.pi * fc / SR)
    q = 0.15  # 小さいほど共鳴が強い
    low = band = 0.0
    out = np.empty(n)
    for i in range(n):
        low += f[i] * band
        high = noise[i] - low - q * band
        band += f[i] * high
        out[i] = band
    out = _lowpass(out, 0.5)
    amp = np.clip(t / 0.2, 0, 1) * (0.5 + 0.5 * np.cos(np.pi * np.clip((t - 0.45) / 0.25, 0, 1)))
    return _reverb(out * amp, tail=0.15, mix=0.25)


def _calib_ok():
    return _reverb(_mix([
        (0.0, _fm_bell(_note(79), 0.6, ratio=1.4, index=1.2, decay=6.0)),    # G5
        (0.15, _fm_bell(_note(84), 0.6, ratio=1.4, index=1.2, decay=6.0)),   # C6
    ]), tail=0.25)


def _reset():
    return _reverb(_mix([
        (k * 0.08, _fm_bell(_note(n), 0.2, ratio=1.0, index=0.5, decay=14, attack=0.008))
        for k, n in enumerate([67, 64, 60])  # G4 E4 C4
    ]), tail=0.15, mix=0.25)


def _finish(x, fade_in=0.002, fade_out=0.03):
    """DC 除去 → 端のフェード → ピーク -3dBFS に正規化 → int16。"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n_in, n_out = int(SR * fade_in), int(SR * fade_out)
    x[:n_in] *= np.linspace(0, 1, n_in)
    x[-n_out:] *= np.linspace(1, 0, n_out)
    peak = np.max(np.abs(x))
    if peak > 0:
        x = x / peak * PEAK
    return np.round(x * 32767).astype(np.int16)


def synth(name):
    """name の音を int16 配列 (44.1kHz mono) で返す。"""
    raw = {
        "hit10": _hit10, "hit50": _hit50, "hit100": _hit100,
        "calib_start": _calib_start, "calib_ok": _calib_ok, "reset": _reset,
    }[name]()
    return _finish(raw)


def _wav_bytes(samples):
    import io
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(samples.tobytes())
    return bio.getvalue()


# ---- キャッシュと再生 ------------------------------------------------------

def _default_dirs():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return [os.path.join(root, "sounds"), os.path.join(tempfile.gettempdir(), "syateki_sounds")]


def _write_all(d):
    os.makedirs(d, exist_ok=True)
    paths = {}
    for name in NAMES:
        p = os.path.join(d, name + ".wav")
        data = _wav_bytes(synth(name))
        try:
            with open(p, "rb") as f:
                same = f.read() == data
        except OSError:
            same = False
        if not same:
            with open(p, "wb") as f:
                f.write(data)
        paths[name] = p
    return paths


def _find_player():
    if winsound:
        return "winsound"
    for cmd in ("paplay", "aplay"):
        exe = shutil.which(cmd)
        if exe:
            return [exe] if cmd == "paplay" else [exe, "-q"]
    return None


def init(cache_dir=None):
    """WAV を合成して cache_dir (既定: <repo>/sounds、だめなら一時フォルダ) に書き出す。"""
    global _paths, _player
    dirs = [cache_dir] if cache_dir else _default_dirs()
    for d in dirs:
        try:
            paths = _write_all(d)
            break
        except Exception:
            continue
    else:
        paths = {}
    with _lock:
        _paths = paths
        _player = _find_player()
    _start_worker()


def _start_worker():
    global _worker
    with _lock:
        if _worker is None or not _worker.is_alive():
            _worker = threading.Thread(target=_run, name="sound", daemon=True)
            _worker.start()


def _run():
    global _proc
    while True:
        name = _queue.get()
        while not _queue.empty():  # 溜まっていたら最新だけ鳴らす
            name = _queue.get_nowait()
        try:
            if not _paths:
                init()
            path = _paths.get(name)
            if not path or not _player:
                continue
            if _player == "winsound":
                winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            else:
                if _proc is not None and _proc.poll() is None:
                    _proc.terminate()  # 上書き
                _proc = subprocess.Popen(_player + [path], stdin=subprocess.DEVNULL,
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


def play(name):
    """非同期で鳴らす。ブロックしない。失敗しても例外を出さない。"""
    try:
        _queue.put_nowait(name)
        if _worker is None:
            _start_worker()
    except Exception:
        pass


_SCORE_NAMES = {100: "hit100", 50: "hit50", 10: "hit10", -1: "calib_ok"}


def play_sound(score):
    """旧 API 互換。score: 100/50/10、-1 はキャリブ完了。"""
    name = _SCORE_NAMES.get(score)
    if name:
        play(name)
