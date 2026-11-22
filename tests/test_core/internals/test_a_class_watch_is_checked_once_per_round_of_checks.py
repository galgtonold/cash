"""A class's watch is checked once per round of checks.

A key build asks for the watch of each class it meets (`class_memo.proof_for`)
again and again: about 30 times for an lmfit fit's hit, each time reading
every cell and class tag the watch holds (450 of them for lmfit's
``ModelResult``) and its probed data. Within one round of checks
(`hit_memo.probe_round`: a key build, building or checking a trace) only
cash's code runs, so a watch that held holds until the world epoch moves
or the class is scanned again. Outside a round, and in the next round,
each check reads it all again.
"""

from __future__ import annotations

import importlib
import sys
import textwrap

import pytest

from cash import Cash
from cash.decorator import class_memo, hit_memo

pytestmark = [
    pytest.mark.core,
    pytest.mark.skipif(not class_memo.ENABLED, reason="the class memo needs the C extension's world watch"),
]

_MODULE = """
TABLE = [1, 2, 3]


class First:
    def run(self, x):
        return x + TABLE[0]
"""


@pytest.fixture
def mod(tmp_path, monkeypatch):
    name = f"round_watch_mod_{tmp_path.name.replace('-', '_')}"
    (tmp_path / f"{name}.py").write_text(textwrap.dedent(_MODULE), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


@pytest.fixture
def full_checks(monkeypatch):
    """How many times a watch's cells and tags were read whole."""
    checks = []
    real = class_memo._strong

    def counting(flat):
        checks.append(1)
        return real(flat)

    monkeypatch.setattr(class_memo, "_strong", counting)
    return checks


def test_within_a_round_a_watch_that_held_is_not_read_again(mod, full_checks):
    proof = class_memo.proof_for(mod.First, "digest")
    assert proof is not None
    del full_checks[:]
    with hit_memo.probe_round():
        for _ in range(3):
            assert class_memo.proof_for(mod.First, "digest") is proof
    assert len(full_checks) == 2, "read again within one round"
    with hit_memo.probe_round():
        assert class_memo.proof_for(mod.First, "digest") is proof
    assert len(full_checks) == 4, "the next round took the last one's word"
    assert class_memo.proof_for(mod.First, "digest") is proof
    assert class_memo.proof_for(mod.First, "digest") is proof
    assert len(full_checks) == 8, "outside a round a check took another's word"


def _first_proof(cls):
    """The class's watch, asked again if a write elsewhere in the process
    (another thread of the test worker) moved the world epoch while it was
    scanned: that scan gives no proof, and the next one does. The world is
    not held still here, as this test needs it to move."""
    for _ in range(5):
        proof = class_memo.proof_for(cls, "digest")
        if proof is not None:
            return proof
    return None


def test_a_moved_world_is_read_again_within_the_round(mod):
    proof = _first_proof(mod.First)
    assert proof is not None
    with hit_memo.probe_round():
        assert class_memo.proof_for(mod.First, "digest") is proof
        stamp = proof.stamp

        def run(self, x):
            return x - 1

        mod.First.run = run
        assert class_memo.proof_for(mod.First, "digest") is proof
        assert proof.stamp != stamp, "a moved world took the round's word: the class was not watched again"


def test_an_edit_between_calls_reaches_the_next_call(mod, tmp_path):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ran = []

    @c.cache
    def apply(obj, x):
        ran.append(x)
        return obj.run(x)

    first = mod.First()
    for _ in range(3):
        assert apply(first, 10) == 11
    mod.TABLE[0] = 5
    assert apply(first, 10) == 15

    def run(self, x):
        return x * 2

    mod.First.run = run
    assert apply(first, 10) == 20
    assert ran == [10, 10, 10]
