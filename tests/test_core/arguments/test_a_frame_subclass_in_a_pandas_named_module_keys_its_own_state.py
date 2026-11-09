"""A user's subclass of a pandas frame keys on its own state too, even in a
module whose name starts with ``pandas``.

Picked by module prefix, a ``DataFrame(pd.DataFrame)`` in the user's
``pandas_frames.py`` was claimed by pandas' own content hasher, which keys
a frame's columns and index only: its ``tag`` (an attribute listed in
``_metadata``, carried through pandas' operations) was left out of the key,
and a call on the frame tagged ``"two"`` was served the answer for ``"one"``.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

SOURCE = """
import pandas as pd


class DataFrame(pd.DataFrame):
    _metadata = ["tag"]

    @property
    def _constructor(self):
        return DataFrame


class Series(pd.Series):
    _metadata = ["tag"]

    @property
    def _constructor(self):
        return Series
"""


@pytest.fixture
def pandas_frames(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    (tmp_path / "pandas_frames.py").write_text(SOURCE)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "pandas_frames", raising=False)
    yield importlib.import_module("pandas_frames")
    sys.modules.pop("pandas_frames", None)


@pytest.mark.parametrize("name", ["DataFrame", "Series"])
def test_its_own_state_is_in_the_key(cash_instance, pandas_frames, name):
    kind = getattr(pandas_frames, name)

    def make(tag, k=0):
        value = kind({"a": np.arange(2000) + k}) if name == "DataFrame" else kind(np.arange(2000) + k)
        value.tag = tag
        return value

    calls = []

    @cash_instance.cache
    def tagged(x):
        calls.append(x.tag)
        return x.tag

    assert tagged(make("one")) == "one"
    assert tagged(make("one")) == "one"
    assert tagged(make("two")) == "two"
    assert tagged(make("one", 1)) == "one"
    assert calls == ["one", "two", "one"]
