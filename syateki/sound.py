"""効果音。numpy で合成した WAV (FM ベル + 短い残響) をキャッシュし、ゲームループを止めずに鳴らす。

    init()            # 起動時に1回。WAV を書き出す
    play("hit100")    # すぐ返る。失敗しても例外を出さない

名前: hit100 / hit50 / hit10 / calib_ok / calib_start / reset / hit100_fever /
      bonus_start / bonus_end
キャッシュ済み WAV が揃っていて SOUND_VERSION が一致すれば合成を省く (音を変えたら上げる)。
再生: Windows は winsound.PlaySound (SND_ASYNC)、Linux は aplay / paplay。
どちらも無ければ無音。鳴っている途中の次の音は上書き。

BGM (ボーナス中):
    bgm_start()       # <repo>/bgm/ の最初の .mp3/.wav を config.BGM_START_SEC 秒から流す
    bgm_stop()        # フェードアウトして止める
曲ファイルがなければ合成した bonus_bgm.wav を使う。Windows は MCI (winmm) で鳴らすので
効果音 (PlaySound) と同時に鳴る。Linux は ffplay (なければ paplay、wav のみ)。
"""
import os
import platform
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import wave

import numpy as np

SR = 44100
PEAK = 10 ** (-3 / 20)  # -3 dBFS
NAMES = ("hit100", "hit50", "hit10", "calib_ok", "calib_start", "reset", "hit100_fever",
         "bonus_start", "bonus_end", "bonus_bgm")
SOUND_VERSION = "2"

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


def _chip(freq, dur, kind="pulse", duty=0.25, decay=2.0, sustain=0.75, vib_from=None, attack=0.003):
    """帯域制限したパルス波/三角波/鋸波 (倍音を足して作るのでエイリアスが出ない)。
    vib_from 秒以降に軽いビブラートを掛ける。"""
    t = _t(dur)
    f = np.full(len(t), float(freq))
    if vib_from is not None:
        depth = 0.012 * np.clip((t - vib_from) / 0.08, 0, 1)
        f *= 1 + depth * np.sin(2 * np.pi * 6.5 * t)
    phase = 2 * np.pi * np.cumsum(f) / SR
    s = np.zeros(len(t))
    for k in range(1, int(16000 / freq) + 1):
        if kind == "pulse":
            s += 2 / (k * np.pi) * np.sin(k * np.pi * duty) * np.cos(k * phase)
        elif kind == "saw":  # 高域を少し丸めた鋸波 (ブラス風)
            s += 2 / (k * np.pi) * (-1) ** (k + 1) * np.sin(k * phase) * np.exp(-k * freq / 5000)
        elif k % 2:  # 三角波
            s += 8 / np.pi ** 2 * (-1) ** ((k - 1) // 2) / k ** 2 * np.sin(k * phase)
    amp = sustain + (1 - sustain) * np.exp(-25 * t)  # 頭だけ少し強い
    return _shape(s, t, attack, decay, release=0.015) * amp


def _reset():
    """パチスロ風のオリジナルジングル (BPM 165、16分の跳ねるリズム)。"""
    s16 = 60 / 165 / 4
    # (開始16分, 長さ16分, MIDI): デン・デデ・デン・デレ → 決め
    melody = [(0, 2, 67), (2, 1, 67), (3, 1, 69), (5, 2, 72), (7, 1, 74), (8, 1, 76)]
    parts = []
    for st, ln, n in melody:
        parts.append((st * s16, 0.5 * _chip(_note(n), ln * s16 * 0.9, duty=0.25)))
    # 決めの1音: C6 + 下に E5・G5 (デューティ50%)、ビブラートで伸ばして減衰
    end = 10 * s16
    parts.append((end, 0.5 * _chip(_note(84), 0.5, duty=0.25, decay=4.0, vib_from=0.1)))
    for n in (76, 79):
        parts.append((end, 0.2 * _chip(_note(n), 0.5, duty=0.5, decay=4.0, vib_from=0.1)))
    # 三角波のベース: C3 / G2 / C3
    for st, ln, n in [(0, 4, 48), (5, 4, 43), (10, 5, 48)]:
        parts.append((st * s16, 0.6 * _chip(_note(n), ln * s16, kind="tri", decay=3.0 if st == 10 else 1.0)))
    # コイン風の高音アクセント
    parts.append((end + 0.02, 0.1 * _chip(_note(100), 0.06, duty=0.5, decay=20)))
    parts.append((end + 0.08, 0.1 * _chip(_note(105), 0.18, duty=0.5, decay=18)))
    return _mix(parts)


def _brass(freq, dur, decay=1.5, vib_from=None):
    """デチューンした鋸波2本 + 1オクターブ上のパルス (リード)。"""
    a = _chip(freq * 1.0035, dur, kind="saw", decay=decay, attack=0.015, vib_from=vib_from)
    b = _chip(freq / 1.0035, dur, kind="saw", decay=decay, attack=0.015, vib_from=vib_from)
    lead = _chip(freq * 2, dur, duty=0.25, decay=decay, vib_from=vib_from)
    return 0.5 * (a + b) + 0.25 * lead


def _snare(dur=0.05, decay=60.0, seed=0):
    t = _t(dur)
    noise = np.random.default_rng(seed).standard_normal(len(t))
    return _shape(noise, t, 0.001, decay, release=0.005)


def _hit100_fever():
    """fever モードの大当たりファンファーレ (オリジナル)。約3.8秒。
    キラキラ上昇 → ドラムロール → ブラスの勝利フレーズ → 長いキメ。"""
    parts = []
    # 1) キラキラの上昇グリッサンド: C メジャースケールで C5 → C8
    scale = [0, 2, 4, 5, 7, 9, 11]
    gl = [72 + 12 * (i // 7) + scale[i % 7] for i in range(22)]
    for i, n in enumerate(gl):
        parts.append((i * 0.03, (0.3 + 0.2 * i / len(gl)) * _chip(_note(n), 0.12, duty=0.5, decay=25)))
    # 2) ドラムロール: だんだん速く・大きく (0.55〜1.3秒)
    t, k = 0.55, 0
    while t < 1.28:
        prog = (t - 0.55) / 0.75
        parts.append((t, (0.1 + 0.3 * prog) * _snare(seed=k)))
        t += 0.06 - 0.03 * prog
        k += 1
    parts.append((0.55, 0.5 * _chip(_note(36), 0.75, kind="tri", decay=0.5, attack=0.4)))  # 下でうなるベース
    # 3) ブラスの勝利フレーズ (BPM 160)
    s16 = 60 / 160 / 4
    t0 = 1.3
    phrase = [(0, 2, 72), (2, 1, 72), (3, 1, 74), (4, 2, 76), (6, 2, 79),
              (8, 3, 77), (11, 1, 76), (12, 2, 74), (14, 2, 79)]
    for st, ln, n in phrase:
        parts.append((t0 + st * s16, 0.6 * _brass(_note(n), ln * s16 * 0.92)))
    for st, n in [(0, 48), (4, 48), (8, 53), (12, 55)]:  # ベース C C F G
        for h in (0, 2):
            parts.append((t0 + (st + h) * s16, 0.5 * _chip(_note(n), s16 * 1.8, kind="tri", decay=4)))
    for st in range(0, 16, 2):  # 8分のスネア
        parts.append((t0 + st * s16, (0.12 if st % 4 else 0.2) * _snare(seed=100 + st)))
    # 4) キメ: C メジャー和音を伸ばしてビブラート、ドンと鳴らして減衰
    t1 = t0 + 16 * s16
    for n, g in [(72, 0.3), (76, 0.25), (79, 0.25), (84, 0.35)]:
        parts.append((t1, g * _brass(_note(n), 0.85, decay=2.5, vib_from=0.2)))
    parts.append((t1, 0.6 * _chip(_note(36), 0.85, kind="tri", decay=3.0)))
    parts.append((t1, 0.5 * _sub(0.6, decay=6.0)))
    parts.append((t1, 0.15 * _snare(0.2, decay=15, seed=999)))
    rng = np.random.default_rng(5)
    for i in range(8):  # 余韻のキラキラ
        n = int(rng.choice([96, 100, 103, 108]))
        parts.append((t1 + 0.05 + i * 0.11, 0.06 * _chip(_note(n), 0.1, duty=0.5, decay=30)))
    return _reverb(_mix(parts), tail=0.15, mix=0.2)


def _svf(x, fc, q, mode="band"):
    """状態変数フィルター。fc は定数か各サンプルの配列。q 小さいほど共鳴が強い。"""
    fc = np.broadcast_to(np.asarray(fc, dtype=float), x.shape)
    f = 2 * np.sin(np.pi * np.minimum(fc, SR / 6) / SR)
    low = band = 0.0
    out = np.empty_like(x)
    for i in range(len(x)):
        low += f[i] * band
        band += f[i] * (x[i] - low - q * band)
        out[i] = band if mode == "band" else low
    return out


def _gong(f0, dur, decay=1.6):
    """銅鑼: 不協和な倍音を少しずつ違う速さで減衰させ、頭にノイズの打撃。"""
    t = _t(dur)
    s = np.zeros(len(t))
    for r, a, d in [(1.0, 1.0, 1.0), (1.47, 0.7, 1.3), (2.09, 0.6, 1.6), (2.56, 0.5, 2.0),
                    (2.76, 0.45, 2.2), (3.42, 0.35, 2.8), (4.17, 0.25, 3.5), (5.2, 0.18, 4.5),
                    (6.73, 0.12, 5.5), (8.31, 0.08, 7.0)]:
        bloom = 1 + 0.004 * (1 - np.exp(-3 * t))  # 鳴った後に少しピッチが上がる
        s += a * np.sin(2 * np.pi * f0 * r * t * bloom) * np.exp(-decay * d * t)
    noise = np.random.default_rng(11).standard_normal(len(t))
    s += 0.6 * _lowpass(noise, 0.3) * np.exp(-25 * t)
    return _shape(s, t, 0.004, 0.0, release=0.1)


def _taiko(gain=1.0, seed=0):
    """和太鼓: ピッチが落ちる低いサイン + 短いこもったノイズ。"""
    dur = 0.45
    t = _t(dur)
    f = 50 + 45 * np.exp(-25 * t)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-9 * t)
    noise = _lowpass(np.random.default_rng(seed).standard_normal(len(t)), 0.15) * np.exp(-40 * t)
    return gain * _shape(body + 0.8 * noise, t, 0.002, 0.0, release=0.05)


def _roar(dur):
    """龍の咆哮: 唸る鋸波 + 低域ノイズを、下がっていくフォルマント2本に通す。"""
    t = _t(dur)
    pitch = 95 * (0.6 ** (t / dur))
    growl = 1 + 0.5 * np.sin(2 * np.pi * 27 * t)
    phase = 2 * np.pi * np.cumsum(pitch) / SR
    src = sum(np.sin(k * phase) / k for k in range(1, 60)) * growl
    src += 0.8 * _lowpass(np.random.default_rng(3).standard_normal(len(t)), 0.2)
    slide = 0.6 ** (t / dur)
    out = _svf(src, 650 * slide, 0.25) + 0.7 * _svf(src, 1100 * slide, 0.3)
    amp = np.clip(t / 0.25, 0, 1) * (0.5 + 0.5 * np.cos(np.pi * np.clip((t - dur + 0.35) / 0.35, 0, 1)))
    return out * amp


def _fue(freq, dur, decay=1.0, vib_from=0.08):
    """笛: サイン + 少しの倍音 + 息のノイズ、ビブラート。"""
    t = _t(dur)
    f = freq * (1 + 0.01 * np.clip((t - vib_from) / 0.1, 0, 1) * np.sin(2 * np.pi * 6 * t))
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) + 0.15 * np.sin(2 * ph) + 0.06 * np.sin(3 * ph)
    breath = _svf(np.random.default_rng(int(freq)).standard_normal(len(t)), freq, 0.5)
    s += 0.25 * breath * (0.4 + 0.6 * np.exp(-20 * t))
    return _shape(s, t, 0.02, decay, release=0.03)


def _bonus_start():
    """ボーナス開始 (オリジナル)。約2.1秒 (BGM と重なるので短め)。
    銅鑼 → 和太鼓の連打 + 龍の咆哮 → ヨナ抜き音階の笛の上昇 → 「×2」のキメ (ドン、ドーン)。"""
    parts = [(0.0, 0.8 * _gong(131, 2.0))]
    # 和太鼓: ドン・ドン・ドドドドン (だんだん強く)
    hits = [0.3, 0.5, 0.7, 0.78, 0.86, 0.94]
    for i, h in enumerate(hits):
        parts.append((h, _taiko(0.5 + 0.4 * i / len(hits), seed=i)))
    parts.append((0.35, 0.22 * _roar(0.75)))
    # ヨナ抜き (C D E G A) の上昇: C5 D5 E5 G5 A5 C6 D6 E6
    for i, n in enumerate([72, 74, 76, 79, 81, 84, 86, 88]):
        last = i == 7
        parts.append((1.0 + i * 0.05, 0.7 * _fue(_note(n), 0.25 if last else 0.065, decay=4.0 if last else 0.0)))
    # キメ: 2発 (×2)。太鼓 + ブラスの C・G・C、2発目は伸ばす
    for k, st in enumerate([1.45, 1.62]):
        ln = 0.1 if k == 0 else 0.35
        parts.append((st, 1.3 * _taiko(seed=50 + k)))
        for n in (60, 67, 72):
            parts.append((st, 0.4 * _brass(_note(n), ln, decay=3.5 if k else 0.0, vib_from=0.12 if k else None)))
    return _reverb(_mix(parts), tail=0.12, mix=0.25)


def _bonus_end():
    """ボーナス終了: 炎が消えるような下降スイープ (フィルターノイズ + 音程) と鐘一打。"""
    dur = 0.75
    t = _t(dur)
    noise = np.random.default_rng(21).standard_normal(len(t))
    fc = 3500 * (150 / 3500) ** (t / dur)
    hiss = _svf(_svf(noise, fc, 0.7, mode="low"), fc, 0.9, mode="low")  # 24dB/oct で高域を絞っていく
    tone = np.sin(2 * np.pi * np.cumsum(1400 * (200 / 1400) ** (t / dur)) / SR)
    amp = np.clip(t / 0.03, 0, 1) * np.exp(-2.5 * t)
    sweep = _shape((0.6 * hiss + 0.3 * tone) * amp, t, 0.01, 0.0, release=0.1)
    bell = _fm_bell(_note(67), 0.6, ratio=1.4, index=1.0, decay=5.0)  # G4
    return _reverb(_mix([(0.0, sweep), (0.3, 0.45 * bell)]), tail=0.1, mix=0.25)


def _kick():
    t = _t(0.25)
    f = 45 + 110 * np.exp(-30 * t)
    return _shape(np.sin(2 * np.pi * np.cumsum(f) / SR), t, 0.001, 14.0, release=0.03)


def _hat(seed, decay=70.0):
    t = _t(0.05)
    n = np.random.default_rng(seed).standard_normal(len(t) + 2)
    return _shape(np.diff(n, 2) / 4, t, 0.0005, decay, release=0.005)  # 2回差分で高域だけ


def _bonus_bgm():
    """ボーナス中の BGM (曲ファイルがないときの代わり、オリジナル)。
    BPM 128、12拍 (約5.6秒)。コードは 2拍ずつ I-V-vi-IV-I-V (C G Am F C G)。"""
    beat = 60 / 128
    e8 = beat / 2
    nbeat = 12
    total = nbeat * beat
    parts = []
    # キック (4つ打ち)・クラップ (2・4拍)・ハイハット (裏の8分)
    for b in range(nbeat):
        parts.append((b * beat, 0.65 * _kick()))
        if b % 2:
            parts.append((b * beat, 0.3 * _lowpass(_snare(0.12, decay=30, seed=b), 0.5)))
        parts.append((b * beat + e8, 0.12 * _hat(b)))
        parts.append((b * beat, 0.05 * _hat(100 + b, decay=120)))
    chords = [(48, [64, 67, 72]), (43, [62, 67, 71]), (45, [64, 69, 72]),
              (41, [65, 69, 72]), (48, [64, 67, 72]), (43, [62, 67, 71])]
    for c, (root, tones) in enumerate(chords):
        st = c * 2 * beat
        # ベース: 8分で root とオクターブ上を行き来
        for i in range(4):
            n = root if i % 2 == 0 else root + 12
            parts.append((st + i * e8, 0.35 * _chip(_note(n), e8 * 0.8, duty=0.5, decay=6)))
        # コード: 鋸波を裏拍で刻む
        for i in range(4):
            for n in tones:
                parts.append((st + i * e8 + e8 / 2 * (i % 2), 0.09 * _chip(_note(n), e8 * 0.55, kind="saw", decay=5)))
    # リード (矩形波 25%): (開始8分, 長さ8分, MIDI)
    lead = [(0, 1, 76), (1, 1, 79), (2, 1, 84), (3, 1, 79),
            (4, 2, 83), (6, 1, 81), (7, 1, 79),
            (8, 1, 76), (9, 1, 79), (10, 2, 81),
            (12, 1, 77), (13, 1, 76), (14, 2, 72),
            (16, 1, 76), (17, 1, 79), (18, 1, 84), (19, 1, 86),
            (20, 2, 86), (22, 2, 83)]
    for st, ln, n in lead:
        parts.append((st * e8, 0.25 * _chip(_note(n), ln * e8 * 0.85, duty=0.25, decay=2.0, vib_from=0.12)))
    buf = _mix(parts)[: int(SR * total)]
    return np.concatenate([buf, np.zeros(int(SR * total) - len(buf))])


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
        "hit100_fever": _hit100_fever, "bonus_start": _bonus_start, "bonus_end": _bonus_end,
        "bonus_bgm": _bonus_bgm,
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
    paths = {name: os.path.join(d, name + ".wav") for name in NAMES}
    stamp = os.path.join(d, "VERSION")
    try:
        with open(stamp) as f:
            cached = f.read().strip() == SOUND_VERSION
    except OSError:
        cached = False
    if cached and all(os.path.isfile(p) for p in paths.values()):
        return paths
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
    with open(stamp, "w") as f:
        f.write(SOUND_VERSION)
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


# ---- BGM ------------------------------------------------------------------

BGM_EXTS = (".mp3", ".wav")
BGM_VOLUME = 0.6  # 効果音より少し小さく (0〜1)
BGM_FADE_SEC = 0.4

_bgm_queue = queue.Queue()
_bgm_worker = None
_bgm_proc = None
_bgm_mci_open = False


def bgm_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bgm")


def find_bgm_file(directory=None):
    """directory (既定: <repo>/bgm) にある最初の .mp3/.wav (名前順)。なければ None。"""
    d = directory or bgm_dir()
    try:
        names = sorted(n for n in os.listdir(d) if n.lower().endswith(BGM_EXTS))
    except OSError:
        return None
    for n in names:
        p = os.path.join(d, n)
        if os.path.isfile(p):
            return p
    return None


def _bgm_start_sec():
    try:
        from . import config
        return float(getattr(config, "BGM_START_SEC", 0) or 0)
    except Exception:
        return 0.0


def resolve_bgm(directory=None, start_sec=None):
    """(path, start_sec) を返す。曲ファイルがなければ合成 BGM (開始 0 秒)。"""
    p = find_bgm_file(directory)
    if p:
        return p, _bgm_start_sec() if start_sec is None else float(start_sec)
    if not _paths:
        init()
    return _paths.get("bonus_bgm"), 0.0


def _bgm_linux_cmd(path, start_sec):
    """Linux 用の再生コマンド。使えるものがなければ None。"""
    ffplay = shutil.which("ffplay")
    if ffplay:
        return [ffplay, "-nodisp", "-autoexit", "-loglevel", "quiet",
                "-volume", str(int(100 * BGM_VOLUME)), "-ss", f"{start_sec:.3f}", path]
    paplay = shutil.which("paplay")
    if paplay and path.lower().endswith(".wav"):  # paplay は mp3 も途中開始も不可
        return [paplay, f"--volume={int(65536 * BGM_VOLUME)}", path]
    return None


def _mci(cmd):
    import ctypes
    buf = ctypes.create_unicode_buffer(256)
    err = ctypes.windll.winmm.mciSendStringW(cmd, buf, 255, 0)
    return err, buf.value


def _bgm_play(path, start_sec):
    global _bgm_proc, _bgm_mci_open
    _bgm_halt(fade=False)
    if not path:
        return
    if platform.system() == "Windows":
        if _mci(f'open "{path}" type mpegvideo alias bgm')[0]:
            return
        _bgm_mci_open = True
        _mci("set bgm time format milliseconds")
        _mci(f"setaudio bgm volume to {int(1000 * BGM_VOLUME)}")
        _mci(f"play bgm from {int(1000 * start_sec)}")
    else:
        cmd = _bgm_linux_cmd(path, start_sec)
        if cmd:
            _bgm_proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)


def _bgm_halt(fade=True):
    """BGM を止める。MCI なら音量を段階的に下げてから止める。Linux は即停止。"""
    global _bgm_proc, _bgm_mci_open
    if _bgm_mci_open:
        if fade:
            steps = 8
            for i in range(steps, 0, -1):
                if not _bgm_queue.empty():  # 次の指示が来たらフェードを切り上げる
                    break
                _mci(f"setaudio bgm volume to {int(1000 * BGM_VOLUME * (i - 1) / steps)}")
                time.sleep(BGM_FADE_SEC / steps)
        _mci("stop bgm")
        _mci("close bgm")
        _bgm_mci_open = False
    if _bgm_proc is not None:
        if _bgm_proc.poll() is None:
            _bgm_proc.terminate()
            try:
                _bgm_proc.wait(timeout=1)
            except Exception:
                _bgm_proc.kill()
        _bgm_proc = None


def _bgm_run():
    while True:
        cmd = _bgm_queue.get()
        try:
            if cmd == "stop":
                _bgm_halt(fade=True)
            else:
                _bgm_play(*resolve_bgm())
        except Exception:
            pass


def _bgm_send(cmd):
    global _bgm_worker
    try:
        with _lock:
            if _bgm_worker is None or not _bgm_worker.is_alive():
                _bgm_worker = threading.Thread(target=_bgm_run, name="bgm", daemon=True)
                _bgm_worker.start()
        _bgm_queue.put_nowait(cmd)
    except Exception:
        pass


def bgm_start():
    """BGM を流し始める。すぐ返る。鳴っていれば頭から流し直す。"""
    _bgm_send("start")


def bgm_stop():
    """BGM をフェードアウトして止める (Linux は即停止)。すぐ返る。"""
    _bgm_send("stop")
