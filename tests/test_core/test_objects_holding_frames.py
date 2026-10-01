"""An object holding frames is keyed frame by frame, not pickled whole.

Pickled whole, an object holding eight 40 MB frames was serialised and
hashed on every call, about 0.7 s for a method returning a row count. Opened
up, each frame goes through its content hasher and the copy-on-write memo,
so a repeat call on an unchanged object checks the frames instead of reading
them, and an edit to any of them is still seen.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cash.decorator import arg_hashing

pytestmark = pytest.mark.core

COW = int(pd.__version__.split(".", 1)[0]) >= 3


class Bundle:
    def __init__(self, n=1000):
        self.name = "sales"
        self.frames = {"a": pd.DataFrame({"x": range(n)}), "b": pd.DataFrame({"y": range(n)})}
        self.extra = [pd.Series(range(n))]


@pytest.mark.skipif(not COW, reason="the memo needs pandas copy-on-write")
def test_a_repeat_call_does_not_read_unchanged_frames_again(disk_cash, monkeypatch):
    @disk_cash.cache
    def total(b):
        return int(b.frames["a"]["x"].sum())

    b = Bundle()
    total(b)
    reads = []
    real = arg_hashing.builtin_hash

    def spy(value):
        if isinstance(value, (pd.DataFrame, pd.Series)):
            reads.append(type(value).__name__)
        return real(value)

    monkeypatch.setattr(arg_hashing, "builtin_hash", spy)
    total(b)
    assert reads == []
    b.frames["a"].loc[0, "x"] = 5
    assert total(b) == 499505
    assert reads == ["DataFrame"]
