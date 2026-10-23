"""A function that draws from an iterator it reaches (``next(src)`` on a
global ``itertools.count()`` or a generator) moves it on each call. Served
from the call cache, a call would hand back the old draw and leave the
iterator where it was, so the next draw would repeat one plain Python already
took: such a call is never stored.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

import cash
from cash.notebook.call_interception import CallSite
from tests._call_cache import make_call_cache
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.fixture
def call_cache(tmp_path):
    return make_call_cache(cash.Cash(cache_dir=str(tmp_path / "cc")))


@pytest.mark.parametrize(
    "made",
    ["itertools.count(1)", "(i for i in range(1, 100))", "iter(range(1, 100))"],
    ids=["count", "generator", "iterator"],
)
def test_each_call_draws_the_next_value(call_cache, tmp_path, monkeypatch, made):
    name = f"_drawing_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(
        f"import itertools, time\nsrc = {made}\n"
        f"def draw(n):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return n * next(src)\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        mod = importlib.import_module(name)
        call_cache.set_sites([CallSite(source="draw(n)", free_names=frozenset({"n"}), occurrence_index=0)])
        cached = call_cache.resolve(mod.draw)
        assert [cached(10), cached(10), next(mod.src)] == [10, 20, 3]
    finally:
        sys.modules.pop(name, None)
