"""Data that many class watches hold is read whole once per key build, and
a user class held by module data is digested once while its world holds.

A module's data no watcher sees (a list, a library's object) is read whole
again before each reuse of a remembered class fold. When a dozen classes
reach one module (lmfit's line shapes, holding four scipy distributions),
each watch read it again: 5 ms of every hit. Within one key build, or one
check of a stored trace, cash's own code runs between the reads, never the
user's body, so each value is read once there. A change between calls is
still seen.

A class a module global holds (``Parameter``, imported into each module of
lmfit) is named by a call's trace and digested again on every check; it is
now remembered while its world holds, as a class the trace names is.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash import Cash, _native
from cash.decorator import class_memo, hit_memo
from cash.decorator.code_args import CodeArgs
from cash.tracking import trace_items

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not class_memo.ENABLED, reason="the class memo needs the C extension's world watch"),
]

_MODULE = """
TABLE = [1, 2, 3]


class First:
    def run(self, x):
        return x + TABLE[0]


class Second:
    def run(self, x):
        return x * TABLE[1]


class Held:
    def run(self, x):
        return x - 1


HOLDER = Held
"""


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"shared_data_mod_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture
def table_reads(mod, monkeypatch):
    """How many times ``TABLE`` was read whole by a probe."""
    reads = []
    real = hit_memo._plain_content

    def counting(v):
        if v is mod.TABLE:
            reads.append(1)
        return real(v)

    monkeypatch.setattr(hit_memo, "_plain_content", counting)
    return reads


@pytest.fixture
def apply(tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ran = []

    @c.cache
    def apply(left, right, x):
        ran.append(x)
        return left.run(x) + right.run(x)

    apply.ran = ran
    return apply


def _probing_watches(mod):
    count = 0
    for cls in (mod.First, mod.Second):
        for purpose in ("carrier", "surface"):
            proof = class_memo._proofs.get((id(cls), purpose))
            family = None if proof is None else proof.family
            if family is not None and any(getattr(read, "value", None) is mod.TABLE for read, _ in family.probes):
                count += 1
    return count


def test_each_key_build_reads_the_shared_data_once(mod, apply, table_reads, monkeypatch):
    first, second = mod.First(), mod.Second()
    for _ in range(3):
        apply(first, second, 10)
    assert _probing_watches(mod) >= 2, "the watches do not probe the shared list: nothing to read once"
    in_fold = []
    real = CodeArgs.fold_code_args

    def fold_code_args(self, *args, **kwargs):
        before = len(table_reads)
        try:
            return real(self, *args, **kwargs)
        finally:
            in_fold.append(len(table_reads) - before)

    monkeypatch.setattr(CodeArgs, "fold_code_args", fold_code_args)
    apply(first, second, 10)
    assert in_fold == [1], "the shared list was read once for each watch holding it"
    assert apply.ran == [10]


def test_a_change_to_the_shared_data_still_reaches_the_key(mod, apply, table_reads):
    first, second = mod.First(), mod.Second()
    for _ in range(3):
        apply(first, second, 10)
    key = apply.explain(first, second, 10).cache_key
    mod.TABLE[1] = 5
    assert apply.explain(first, second, 10).cache_key != key
    assert apply(first, second, 10) == 11 + 50
    assert apply.ran == [10, 10]


def test_outside_a_key_build_each_check_reads_it_again(mod, apply, table_reads):
    first, second = mod.First(), mod.Second()
    for _ in range(3):
        apply(first, second, 10)
    proof = class_memo._proofs.get((id(mod.First), "carrier"))
    assert proof is not None
    del table_reads[:]
    assert proof.holds() and proof.holds()
    assert len(table_reads) == 2


def test_a_class_module_data_holds_is_digested_once_while_its_world_holds(mod, monkeypatch):
    digests = []
    real = trace_items._class_digest_now

    def counting(cls, *args, **kwargs):
        if cls is mod.Held:
            digests.append(1)
        return real(cls, *args, **kwargs)

    monkeypatch.setattr(trace_items, "_class_digest_now", counting)
    # The world epoch is the process's: anything else in it writing a world
    # dict (a thread an earlier test left, such as a cache writer) moves it.
    # A watch taken while it moved is not trusted, so the class is watched
    # and digested again on the next call (`class_memo._Proof.scan`): one
    # more digest per move, and none while it holds.
    epoch = _native.cwatch.world_epoch()
    first = trace_items.value_digest(mod.HOLDER)
    assert trace_items.value_digest(mod.HOLDER) == first
    assert trace_items.value_digest(mod.HOLDER) == first
    moves = _native.cwatch.world_epoch() - epoch
    assert len(digests) <= 1 + moves, f"digested {len(digests)} times while the world moved {moves} times"
    with monkeypatch.context() as m:
        m.setattr(class_memo, "ENABLED", False)
        assert trace_items.value_digest(mod.HOLDER) == first, "remembered, it is not the digest read now"

    def run(self, x):
        return x - 2

    mod.Held.run.__code__ = run.__code__
    assert trace_items.value_digest(mod.HOLDER) != first
