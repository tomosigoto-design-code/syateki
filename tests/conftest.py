import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest


@pytest.fixture(autouse=True)
def _silence(monkeypatch):
    """テスト中は音を鳴らさない (WAV の書き出しもしない)。"""
    from syateki import sound
    monkeypatch.setattr(sound, "play", lambda name: None)
