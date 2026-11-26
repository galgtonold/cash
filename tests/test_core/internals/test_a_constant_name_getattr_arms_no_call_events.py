"""``getattr(obj, "name")`` reads ``obj.name``: the trace recorder arms no
CALL events for it.

backtesting.py's ``_Array``, an ndarray subclass, copies two attributes in
``__array_finalize__`` with ``getattr(obj, "_name", None)`` and
``getattr(obj, "_opts", {})`` on every slice. The recorder armed CALL events
on any code of the user's that names ``getattr``, so a traced miss paid a
callback per slice (two million over a dozen backtests), read each call's
name argument, and kept every sliced array alive in the unit until the call
ended: the body ran 1.85 times as long as plain. Only a ``getattr`` whose
name is not a constant, and an import by a name, now arm CALL events; a
constant name is the attribute read it spells, keyed as ``obj.name`` is,
and an edit behind it still misses.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
]


def _code(source: str):
    ns: dict = {}
    exec(compile(textwrap.dedent(source), "<test>", "exec"), ns)
    return ns["f"].__code__


@pytest.mark.parametrize(
    ("source", "by_name"),
    [
        ("def f(obj):\n    return getattr(obj, '_opts', {})\n", False),
        ("def f(obj):\n    return getattr(obj, '_name', None), getattr(obj, '_tags', [])\n", False),
        ("def f(obj):\n    return getattr(obj, 'run')(1)\n", False),
        ("def f(obj):\n    return getattr(getattr(obj, 'a'), 'b', ())\n", False),
        ("def f(obj, name):\n    return getattr(obj, name)\n", True),
        ("def f(obj, name):\n    return getattr(obj, 'a' + name, None)\n", True),
        ("def f(obj):\n    get = getattr\n    return get(obj, 'a')\n", True),
        ("def f(name):\n    import importlib\n    return importlib.import_module(name)\n", True),
    ],
)
def test_only_a_name_known_at_run_time_is_a_lookup(source, by_name):
    from cash.decorator.code_trace import looks_up_by_name

    assert looks_up_by_name(_code(source)) is by_name


def test_an_empty_default_does_not_hide_a_constant_name():
    from cash.decorator.code_trace import _plan

    code = _code("def f(obj):\n    return getattr(obj, '_opts', {})\n")
    import dis

    call = next(i for i in dis.get_instructions(code) if i.opname == "CALL")
    assert _plan(code, call.offset) == ("const", "_opts")


ARRAY = """
import gc, os, sys, time, weakref
sys.path.insert(0, os.getcwd())
import numpy as np
import cash
from cash.decorator import code_trace


class Arr(np.ndarray):
    def __new__(cls, a, name=None, **opts):
        obj = np.asarray(a).view(cls)
        obj._opts = opts
        obj._name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self._name = getattr(obj, "_name", None)
            self._opts = getattr(obj, "_opts", {})


class Box:
    pass


def body(n):
    a = Arr(np.arange(2000.0), name="x")
    s = 0.0
    for i in range(n):
        s += float(a[: (i % 1999) + 1][-1])
    # A lookup by a run-time name keeps nothing it looked the name up on.
    name = "value"
    box = Box()
    box.value = 1
    gone = weakref.ref(box)
    getattr(box, name)
    del box
    gc.collect()
    return s, gone() is None


N = 300_000
t = time.perf_counter()
plain = body(N)
plain_s = time.perf_counter() - t
cached = cash.Cash(cache_dir=os.environ["CASH_CACHE_DIR"]).cache(body)
t = time.perf_counter()
got = cached(N)
miss_s = time.perf_counter() - t
assert got == plain, (got, plain)
assert got[1], "the traced call kept an object it looked a name up on alive"
rec = code_trace._RECORDER
events = rec.user_codes.get(Arr.__array_finalize__.__code__)
assert events is not None, "the recorder never saw __array_finalize__ run"
assert not events & sys.monitoring.events.CALL, events
assert events & sys.monitoring.events.PY_START
assert not any(c.co_filename.startswith("<frozen ") for c in rec.user_codes), "frozen stdlib armed"
# Loose: before, the miss ran several times as long as plain.
assert miss_s < 3 * plain_s + 1.0, (miss_s, plain_s)
print("ok", round(plain_s, 3), round(miss_s, 3))
"""


def test_a_slice_heavy_body_pays_no_call_events(tmp_path):
    (tmp_path / "main.py").write_text(ARRAY, encoding="utf-8")
    out = run_python("main.py", cwd=tmp_path).stdout
    assert out.startswith("ok"), out


HELPERS = """
import cfg

def g(x):
    return getattr(cfg, "THRESHOLD", 1) * x

def h(x):
    return x * {factor}
"""

MAIN = """
import os, sys
sys.path.insert(0, os.getcwd())
import cash
import helpers


@cash.cache
def by_name(x, name):
    return getattr(helpers, name)(x)


name = sys.argv[1]
print(by_name.explain(3, name).reason, by_name(3, name))
"""


def _run(tmp_path, name: str) -> list[str]:
    return run_python("main.py", name, cwd=tmp_path).stdout.split()


def _write(tmp_path, threshold: int, factor: int) -> None:
    (tmp_path / "cfg.py").write_text(f"THRESHOLD = {threshold}\n", encoding="utf-8")
    (tmp_path / "helpers.py").write_text(textwrap.dedent(HELPERS.format(factor=factor)), encoding="utf-8")


def test_a_constant_name_read_in_code_picked_at_run_time_still_misses_on_an_edit(tmp_path):
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    _write(tmp_path, threshold=2, factor=10)
    assert _run(tmp_path, "g") == ["no_entry", "6"]
    assert _run(tmp_path, "g") == ["hit", "6"]
    _write(tmp_path, threshold=5, factor=10)
    reason, value = _run(tmp_path, "g")
    assert reason != "hit"
    assert value == "15"


def test_a_dispatch_by_a_run_time_name_still_misses_on_an_edit(tmp_path):
    (tmp_path / "main.py").write_text(MAIN, encoding="utf-8")
    _write(tmp_path, threshold=2, factor=10)
    assert _run(tmp_path, "h") == ["no_entry", "30"]
    assert _run(tmp_path, "h") == ["hit", "30"]
    _write(tmp_path, threshold=2, factor=7)
    reason, value = _run(tmp_path, "h")
    assert reason != "hit"
    assert value == "21"
