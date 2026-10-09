"""A module constant is hashed once per key build, however many call sites
reach it.

Helpers that call helpers of another module (``other.h(x)``) fold the
constants each callee reads, once per call site: a table fifty call sites
reach was hashed fifty times on every hit. A value cannot change while one
key is built, so its digest is kept for the rest of that build, and only
that build: an edit made between two calls still misses
(``test_core/globals_and_closures/test_a_constant_many_call_sites_reach_is_keyed_as_it_is_now.py``).
"""

from __future__ import annotations

import collections
import importlib
import sys
import textwrap

import pytest

from cash.decorator import global_values

SITES = 12

CONSTS = """
TABLE = {"k1": 1, "k2": 2}
SCALE = [3]


def lookup(x):
    return x + TABLE["k1"] + SCALE[0]
"""


def _callers() -> str:
    lines = ["import fold_once_consts\n"]
    for i in range(SITES):
        lines.append(f"def c{i}(x):\n    return fold_once_consts.lookup(x)\n")
    lines.append("def top(x):\n    return " + " + ".join(f"c{i}(x)" for i in range(SITES)) + "\n")
    return "\n\n".join(lines)


@pytest.fixture
def modules(tmp_path, monkeypatch):
    (tmp_path / "fold_once_consts.py").write_text(textwrap.dedent(CONSTS), encoding="utf-8")
    (tmp_path / "fold_once_callers.py").write_text(_callers(), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    yield importlib.import_module("fold_once_consts"), importlib.import_module("fold_once_callers")
    for name in ("fold_once_consts", "fold_once_callers"):
        sys.modules.pop(name, None)


def test_a_constant_many_call_sites_reach_is_hashed_once_per_hit(disk_cash, modules, monkeypatch):
    consts, callers = modules
    cached = disk_cash.cache(callers.top)
    assert cached(1) == SITES * 5
    hashed = collections.Counter()
    real = global_values.GlobalValues.table_digest

    def counting(self, kind, value, compute):
        hashed[id(value)] += 1
        return real(self, kind, value, compute)

    monkeypatch.setattr(global_values.GlobalValues, "table_digest", counting)
    assert cached(1) == SITES * 5
    assert cached.cache_info()["hits"] == 1
    assert hashed[id(consts.TABLE)] == 1, hashed
    assert hashed[id(consts.SCALE)] == 1, hashed
