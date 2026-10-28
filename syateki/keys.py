"""cv2.waitKeyEx のキーコードを名前に変換する。

旧版は waitKey(1) & 0xFF で判定していたため、Linux の矢印キー (0xFF52 など) が
下位8bitで 'R'/'Q'/'S'/'T' に化けていた。
"""

# Windows (仮想キー << 16)
_WIN = {0x260000: "up", 0x280000: "down", 0x250000: "left", 0x270000: "right"}
# Qt (Qt::Key_*)
_QT = {0x1000013: "up", 0x1000015: "down", 0x1000012: "left", 0x1000014: "right"}
# GTK / X11 keysym (上位bitに修飾キーが入るので下位16bitで判定)
_X11 = {0xFF52: "up", 0xFF54: "down", 0xFF51: "left", 0xFF53: "right"}


def decode_key(code):
    """'up'/'down'/'left'/'right'/'esc'/'tab'、または小文字1文字。無ければ None。

    使用中の文字: c m r f q w a s d (角・操作) / i j k l o (着弾補正) / p (演出モード)
    """
    if code is None or code < 0:
        return None
    if code in _WIN:
        return _WIN[code]
    if code in _QT:
        return _QT[code]
    low = code & 0xFFFF
    if low in _X11:
        return _X11[low]
    if low == 27:
        return "esc"
    if low == 9:
        return "tab"
    if 32 < low < 127:
        return chr(low).lower()
    return None
