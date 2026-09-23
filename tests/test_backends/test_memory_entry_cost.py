"""A notebook entry in the RAM tier costs what its values cost, not what is
walked around them.

A profile of real notebooks showed:

* Every entry a notebook statement stores is a dict that carries the RNG state,
  and ``random.getstate()`` is a tuple of 625 ints. Sizing it recursed once per
  int, and so did the deepcopy of the entry: in a loop over 1,312 files,
  2,629 stores made 1.7M sizing calls and 2.3M deepcopy calls.
* The upstream simulation reads entries' METADATA, and the RAM tier had no
  metadata read: the base class's ``get()`` deep-copied the value to throw it
  away -- 5.3 s of one cell's 6 s.
"""

from __future__ import annotations

import copy
import random
import sys

import pytest

from cash.backends.memory_backend import InMemoryBackend

pytestmark = [pytest.mark.core]


class _CountsCopies:
    copies = 0

    def __deepcopy__(self, memo):
        type(self).copies += 1
        return _CountsCopies()


def _entry(value=None):
    return {
        "variables": {"d": value if value is not None else [1, 2, 3]},
        "rng_state": {"random": random.getstate()},
        "stdout": "",
    }


def test_reading_metadata_does_not_copy_the_value():
    b = InMemoryBackend()
    b.set("k", _entry(_CountsCopies()))
    _CountsCopies.copies = 0
    meta = b.get_metadata("k")
    assert meta is not None and meta["key"] == "k"
    assert _CountsCopies.copies == 0, "a metadata read deep-copied the value"
    assert b.get_metadata("missing") is None


def test_reading_metadata_counts_as_an_access_like_get():
    """The same stamps `get` leaves: LRU eviction orders on them."""
    b = InMemoryBackend()
    b.set("k", _entry())
    before = b._store["k"][0]["access_count"]
    b.get_metadata("k")
    assert b._store["k"][0]["access_count"] == before + 1


def test_an_rng_state_is_sized_without_a_call_per_int(monkeypatch):
    b = InMemoryBackend()
    calls = []
    real = InMemoryBackend._get_object_size
    monkeypatch.setattr(
        InMemoryBackend, "_get_object_size", lambda self, obj, seen=None: calls.append(1) or real(self, obj, seen)
    )
    b.set("k", _entry())
    assert len(calls) < 50, f"{len(calls)} sizing calls for one entry"
    assert b._store["k"][0]["size"] > 625 * 28, "the ints still count toward the size"


def _deepcopy_calls(fn) -> int:
    """Calls of ``copy.deepcopy`` while *fn* runs, recursion included -- which a
    monkeypatch cannot see: ``_deepcopy_tuple`` binds ``deepcopy`` as a default."""
    code, n = copy.deepcopy.__code__, [0]

    def profile(frame, event, _arg):
        if event == "call" and frame.f_code is code:
            n[0] += 1

    sys.setprofile(profile)
    try:
        fn()
    finally:
        sys.setprofile(None)
    return n[0]


def test_an_rng_state_is_stored_without_a_copy_per_int():
    b = InMemoryBackend()
    calls = _deepcopy_calls(lambda: b.set("k", _entry()))
    assert calls < 50, f"{calls} deepcopy calls to store one entry"
    calls = _deepcopy_calls(lambda: b.get("k"))
    assert calls < 50, f"{calls} deepcopy calls to read one entry back"


def test_the_stored_entry_is_still_a_copy():
    b = InMemoryBackend()
    shared = [1, 2]
    entry = {"variables": {"a": shared, "b": shared, "t": (1, (2, 3))}, "rng_state": {"random": random.getstate()}}
    b.set("k", entry)
    shared.append("the original's")
    got = b.get("k")[1]
    assert got["variables"]["a"] == [1, 2], "the stored entry shares the original's list"
    assert got["variables"]["a"] is got["variables"]["b"], "two names for one list became two lists"
    got["variables"]["a"].append("caller's")
    assert b.get("k")[1]["variables"]["b"] == [1, 2], "a caller's append reached the stored entry"
    assert got["rng_state"]["random"] == entry["rng_state"]["random"]
