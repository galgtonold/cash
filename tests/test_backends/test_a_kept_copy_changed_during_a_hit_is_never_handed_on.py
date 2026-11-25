"""A kept copy its holder changes while another thread's hit looks at it
is not handed on, nor copied for that hit.

The RAM tier hands a hit the copy it kept, or a copy made from it, once
the watchers prove the kept one unchanged. The tier's lock does not stop
the thread that still holds that copy from writing it: a write landing
between the proof and the hand-out reached the other caller, who got a
result holding the first caller's edit. The proof is now made in the same
C call that counts the references or makes the copy.

Each test runs the holder's write at that point: just before the C call a
hit makes, after anything the hit read earlier.
"""

from __future__ import annotations

import pytest

from cash import handed_copies, value_watch
from cash._native import cwatch
from cash.backends import memory_backend
from cash.backends.memory_backend import InMemoryBackend

pytestmark = pytest.mark.skipif(
    not value_watch.enabled() or not hasattr(cwatch, "clone_held"),
    reason="a copy is kept only where the C extension's watchers prove it unchanged",
)


def table():
    return {"words": {str(i): [i, str(i)] for i in range(20000)}, "log": []}


PLAIN = table()


@pytest.fixture
def memory():
    backend = InMemoryBackend()
    backend.set("k", table(), {"size": 1_000_000})
    assert type(backend._store["k"][1]) is memory_backend._Marshalled, "kept as marshal bytes"
    return backend


def _before(monkeypatch, name: str, write) -> list:
    """Run *write* once, just before the hit's next call of the C function
    *name*: where another thread holding the copy gets to run."""
    real = getattr(handed_copies, name)
    ran: list = []

    def call(*args):
        if not ran:
            ran.append(1)
            write()
        return real(*args)

    monkeypatch.setattr(handed_copies, name, call)
    return ran


def test_a_copy_written_by_its_holder_just_before_it_is_copied_is_not_the_source(memory, monkeypatch):
    _meta, held = memory.get("k")
    ran = _before(monkeypatch, "_CLONE_HELD", lambda: held["log"].append("holder's"))

    _meta, other = memory.get("k")

    assert ran, "the hit copied the held copy"
    assert other is not held
    assert other == PLAIN, "another caller's edit came back from the cache"


def test_a_copy_written_and_dropped_just_before_it_is_handed_back_is_not_handed_back(memory, monkeypatch):
    _meta, held = memory.get("k")
    box = [held]
    del held

    def write_and_drop():
        box[0]["log"].append("holder's")
        box.clear()

    ran = _before(monkeypatch, "_HELD_STILL", write_and_drop)

    _meta, other = memory.get("k")

    assert ran
    assert other == PLAIN, "another caller's edit came back from the cache"


def test_a_part_dropped_just_before_the_references_are_counted_is_not_read(memory, monkeypatch):
    """The count reads the addresses recorded when the copy was kept: only
    while every one of them is still an object of the copy."""
    _meta, held = memory.get("k")
    ran = _before(monkeypatch, "_HELD_STILL", lambda: held["words"].clear())

    _meta, other = memory.get("k")

    assert ran
    assert other == PLAIN


def test_the_c_calls_refuse_a_copy_the_watchers_do_not_vouch_for(memory):
    _meta, held = memory.get("k")
    handed = memory._store["k"][1].handed
    kind = value_watch.KIND_HANDED
    assert cwatch.clone_held(handed.holder, 0, kind, handed.token) == PLAIN, "the positive control"
    assert cwatch.clone_held(handed.holder, 0, kind, object()) is None, "another token"
    held["log"].append(1)
    assert cwatch.clone_held(handed.holder, 0, kind, handed.token) is None
    assert (
        cwatch.held_still(handed.holder, 0, handed.objects, handed.weakable, handed.count, kind, handed.token) is None
    )


def test_an_unchanged_copy_is_still_handed_back_and_copied(memory):
    """The positive control for the tests above."""
    _meta, first = memory.get("k")
    _meta, copied = memory.get("k")
    assert copied is not first and copied == PLAIN
    first_id = id(first)
    del first, copied
    _meta, again = memory.get("k")
    assert id(again) == first_id
