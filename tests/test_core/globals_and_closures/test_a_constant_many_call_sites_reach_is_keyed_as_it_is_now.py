"""A module constant that many call sites reach is keyed as it is now.

Its digest is kept for the rest of one key build, so fifty call sites hash
it once; an edit made between two calls, in place included, still misses.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

pytestmark = pytest.mark.core

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


def test_a_constant_changed_between_calls_still_misses(disk_cash, modules):
    consts, callers = modules
    cached = disk_cash.cache(callers.top)
    assert cached(1) == SITES * 5
    consts.TABLE["k1"] = 10  # in place: the same object, a new value
    assert cached(1) == SITES * 14
    consts.SCALE.append(0)  # a change the read does not see still moves the key
    assert cached(1) == SITES * 14
    assert cached.cache_info()["misses"] == 3
