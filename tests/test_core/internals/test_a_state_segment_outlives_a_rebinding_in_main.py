"""A state segment remembered by the state memo (`cash.decorator.state_memo`)
stays served when the script or notebook rebinds one of its own variables,
unless the segment's world reads that namespace.

A script's loop rebinds its variables between calls, and each one whose
name the scanned code names (``OFFSET``, or ``low`` for ``self.low``)
moved the C extension's core epoch: the state memo scanned the function's
world again and folded its globals again on those calls, hits included (a
NiMARE walk: 13 scans in 27 hits). A plain rebinding of
``__main__`` now leaves the memo standing (`cwatch.core_moved_only_in`); a
world that reads ``__main__`` (a function defined there, code naming it)
still sees every change, and so does any other change.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash import Cash
from cash._native import cwatch
from cash.decorator import state_memo
from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not state_memo.ENABLED, reason="the state memo needs the C extension's world watch"),
]

_MODULE = """
import __main__ as _main_module

RATE = 2
OFFSET = 1


def helper(x, low):
    return x * RATE + low


def compute(x):
    return helper(x, OFFSET)


def reads_main(x):
    return x * _main_module.FACTOR
"""


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"state_rebind_mod_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture
def main_ns():
    """``__main__``'s namespace; every name a test binds there is removed after."""
    ns = sys.modules["__main__"].__dict__
    before = set(ns)
    yield ns
    for name in set(ns) - before:
        del ns[name]


def _cached(tmp_path, fn):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    return c.cache(fn)


def _warm(f):
    """Call until the memo serves the segment: the first build is not
    remembered, the second scans. Each call a miss, so that each builds its
    key (a hit while nothing moved is served without one)."""
    for x in range(100, 104):
        f(x)
    reused = state_memo.STATS["reused"]
    f(104)
    assert state_memo.STATS["reused"] > reused, "the state memo never served the segment"


def test_a_plain_rebinding_in_main_keeps_the_segment_served(tmp_path, mod, main_ns):
    compute = _cached(tmp_path, mod.compute)
    _warm(compute)
    key = compute.explain(10).cache_key
    scans, reused = state_memo.STATS["scans"], state_memo.STATS["reused"]
    epoch = cwatch.core_epoch()
    # ``OFFSET`` is a name the function's code names: rebinding it in
    # ``__main__`` moves the core epoch.
    main_ns["OFFSET"] = 1
    main_ns["OFFSET"] = [1, 2]
    assert cwatch.core_epoch() > epoch
    assert compute(10) == 21
    assert state_memo.STATS["scans"] == scans, "the world was scanned anew"
    assert state_memo.STATS["reused"] > reused, "the segment was folded again"
    assert compute.explain(10).cache_key == key


def test_a_change_in_the_world_still_reaches_the_key(tmp_path, mod, main_ns):
    compute = _cached(tmp_path, mod.compute)
    _warm(compute)
    key = compute.explain(10).cache_key
    main_ns["OFFSET"] = 7  # outside the world: the memo stands
    mod.RATE = 3  # inside it: the module the helper reads
    assert compute.explain(10).cache_key != key
    assert compute(10) == 31


def test_a_function_bound_in_main_is_no_plain_rebinding(tmp_path, mod, main_ns):
    compute = _cached(tmp_path, mod.compute)
    _warm(compute)
    reused = state_memo.STATS["reused"]
    main_ns["OFFSET"] = len
    compute(10)
    assert state_memo.STATS["reused"] == reused, "a callable bound in __main__ was taken for a plain rebinding"


def test_code_naming_main_sees_a_rebinding_there(tmp_path, mod, main_ns):
    main_ns["FACTOR"] = 2
    reads_main = _cached(tmp_path, mod.reads_main)
    _warm(reads_main)
    key = reads_main.explain(10).cache_key
    main_ns["FACTOR"] = 5
    assert reads_main.explain(10).cache_key != key
    assert reads_main(10) == 50


_SCRIPT = """
import cash
from cash.decorator import state_memo

c = cash.Cash(register_magic=False)
SCALE = 2
ran = []


@c.cache
def apply(x):
    ran.append(x)
    return x * SCALE


for low in range(4):
    assert apply(10) == 20
key = apply.explain(10).cache_key
SCALE = 4
assert apply.explain(10).cache_key != key, "a global of the function's own namespace did not reach the key"
assert apply(10) == 40
assert len(ran) == 2, ran
print("ok")
"""


def test_a_function_defined_in_the_script_sees_a_rebinding_there(tmp_path):
    """The script's own function reads ``__main__``: every plain rebinding
    there is seen."""
    (tmp_path / "main.py").write_text(_SCRIPT, encoding="utf-8")
    assert run_python("main.py", cwd=tmp_path).stdout.strip().endswith("ok")


def test_the_core_epoch_moved_only_in_the_namespaces_named(tmp_path, mod, main_ns):
    """`cwatch.core_moved_only_in`: true across plain rebindings in the
    namespaces named, false once anything else moved the core epoch."""
    _warm(_cached(tmp_path, mod.compute))  # watches the module and the names its code names
    main = frozenset((id(main_ns),))
    epoch = cwatch.core_epoch()
    assert cwatch.core_moved_only_in(epoch, main)
    main_ns["OFFSET"] = 3
    main_ns["OFFSET"] = 4
    assert cwatch.core_epoch() > epoch
    assert cwatch.core_moved_only_in(epoch, main)
    assert not cwatch.core_moved_only_in(epoch, frozenset())
    main_ns["OFFSET"] = len
    assert not cwatch.core_moved_only_in(epoch, main)
    epoch = cwatch.core_epoch()
    mod.RATE = 5
    assert cwatch.core_epoch() > epoch
    assert not cwatch.core_moved_only_in(epoch, main)
