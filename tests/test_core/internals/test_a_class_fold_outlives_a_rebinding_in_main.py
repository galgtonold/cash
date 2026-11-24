"""A class fold remembered under its watch (`cash.decorator.class_memo`)
stays served when the script or notebook rebinds one of its own
variables, unless the class's world reads that namespace.

A notebook rebinds its variables on every cell, and each one whose name
the scanned code names (``ids`` for ``self.ids`` counts) moved the C
extension's world epoch: an estimator handed to every call was folded
again on each, hits included (NiMARE's kernel, 25 s of 33 s of cash's
time). A plain rebinding of ``__main__`` now leaves the watch standing
(`cwatch.world_moved_only_in`); a class that reads ``__main__`` (defined
there, its code running there or naming it) still sees every change.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash import Cash
from cash._native import cwatch
from cash.decorator import class_memo
from tests._scripts import run_python

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not class_memo.ENABLED, reason="the class memo needs the C extension's world watch"),
]

_MODULE = """
import __main__ as _main_module

RATE = 2


class Kernel:
    def __init__(self, low):
        self.low = low

    def run(self, x):
        return x * RATE + self.low


class ReadsMain:
    def run(self, x):
        return x * _main_module.FACTOR
"""


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"main_rebind_mod_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture
def main_ns(monkeypatch):
    """``__main__``'s namespace; every name a test binds there is removed after."""
    ns = sys.modules["__main__"].__dict__
    before = set(ns)
    yield ns
    for name in set(ns) - before:
        del ns[name]


def _apply(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ran = []

    @c.cache
    def apply(kernel, x):
        ran.append(x)
        return kernel.run(x)

    return apply, ran


def test_a_plain_rebinding_in_main_keeps_the_fold_served(tmp_path, mod, main_ns):
    apply, ran = _apply(tmp_path)
    k = mod.Kernel(1)
    for _ in range(3):
        assert apply(k, 10) == 21
    key = apply.explain(k, 10).cache_key
    scans, served = class_memo.STATS["scans"], class_memo.STATS["served"]
    epoch = cwatch.world_epoch()
    # ``low`` is a name the class's code names (``self.low``): rebinding it
    # in ``__main__`` moves the world epoch.
    main_ns["low"] = 1
    main_ns["low"] = [1, 2]
    assert cwatch.world_epoch() > epoch
    assert apply(k, 10) == 21
    assert len(ran) == 1
    assert class_memo.STATS["scans"] == scans, "the class was watched anew"
    assert class_memo.STATS["served"] > served, "the class was folded again"
    assert apply.explain(k, 10).cache_key == key


def test_a_change_in_the_class_world_still_reaches_the_key(tmp_path, mod, main_ns):
    apply, ran = _apply(tmp_path)
    k = mod.Kernel(1)
    for _ in range(3):
        apply(k, 10)
    key = apply.explain(k, 10).cache_key
    main_ns["low"] = 7  # outside the world: the watch stands
    mod.RATE = 3  # inside it: the module the method reads
    assert apply.explain(k, 10).cache_key != key
    assert apply(k, 10) == 31
    assert len(ran) == 2


def test_a_class_whose_code_names_main_sees_a_rebinding_there(tmp_path, mod, main_ns):
    apply, ran = _apply(tmp_path)
    main_ns["FACTOR"] = 2
    r = mod.ReadsMain()
    for _ in range(3):
        assert apply(r, 10) == 20
    key = apply.explain(r, 10).cache_key
    main_ns["FACTOR"] = 5
    assert apply.explain(r, 10).cache_key != key
    assert apply(r, 10) == 50
    assert len(ran) == 2


_SCRIPT = """
import cash
from cash.decorator import class_memo

c = cash.Cash(register_magic=False)
SCALE = 2


class MainKernel:
    def __init__(self):
        self.low = 0

    def run(self, x):
        return x * SCALE + self.low


ran = []


@c.cache
def apply(kernel, x):
    ran.append(x)
    return kernel.run(x)


k = MainKernel()
for low in range(3):
    assert apply(k, 10) == 20
key = apply.explain(k, 10).cache_key
SCALE = 4
assert apply.explain(k, 10).cache_key != key, "a global of the class's own namespace did not reach the key"
assert apply(k, 10) == 40
assert len(ran) == 2, ran
print("ok")
"""


def test_a_class_defined_in_the_script_sees_a_rebinding_there(tmp_path):
    """The script's own class reads ``__main__``: every rebinding there is
    seen, ``low`` (named by ``self.low``, rebound by the loop) included."""
    (tmp_path / "main.py").write_text(_SCRIPT, encoding="utf-8")
    assert run_python("main.py", cwd=tmp_path).stdout.strip().endswith("ok")


def test_the_world_moved_only_in_the_namespaces_named(tmp_path, mod, main_ns):
    """`cwatch.world_moved_only_in`: true across plain rebindings in the
    namespaces named, false once anything else moved the world."""
    apply, _ran = _apply(tmp_path)
    for _ in range(3):
        apply(mod.Kernel(1), 10)  # watches the module and the names its code names
    main = frozenset((id(main_ns),))
    epoch = cwatch.world_epoch()
    assert cwatch.world_moved_only_in(epoch, main)
    main_ns["low"] = 3
    main_ns["low"] = 4
    assert cwatch.world_epoch() > epoch
    assert cwatch.world_moved_only_in(epoch, main)
    assert not cwatch.world_moved_only_in(epoch, frozenset())
    # A function bound there is no plain rebinding.
    main_ns["low"] = len
    assert not cwatch.world_moved_only_in(epoch, main)
    epoch = cwatch.world_epoch()
    # Nor is one in another namespace.
    mod.RATE = 5
    assert cwatch.world_epoch() > epoch
    assert not cwatch.world_moved_only_in(epoch, main)
