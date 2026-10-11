"""Big lists of tuples and strings in a RAM entry come back as new lists
over the same rows.

``parsed = [tuple(line.split(',')) for line in lines]`` with 300,000 rows was
kept as marshal bytes, so every hit built the 1.2 million objects again: 1.3
s, twice what parsing took. ``srt = sorted(parsed)`` was copied a container at
a time on every hit (0.8 s against 0.2 s to sort). Nothing can write into a
tuple of strings, so a hit now gets a new list over the stored rows: the same
values, a list of its own, in milliseconds.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from cash.backends import InMemoryBackend
from cash.backends.memory_backend import _Marshalled
from cash.notebook.closed_stream import stored_form

N = 6_000


def _rows(n: int = N) -> list:
    return [(str(i), f"name{i % 50}", str(i * 7)) for i in range(n)]


def _entry(variables: dict, json_like: bool = False) -> dict:
    # A notebook entry: the variables beside output and the RNG states.
    # numpy's holds an array, which makes the entry not JSON-like.
    rng = {"random": random.getstate()}
    if not json_like:
        rng["numpy"] = np.random.get_state()
    return {"variables": variables, "stdout": "", "rich_outputs": [], "rng_state": rng}


def _stored_rows(backend: InMemoryBackend, key: str, name: str):
    return backend._store[key][1]["variables"][name]


@pytest.mark.parametrize("json_like", [False, True], ids=["entry-with-arrays", "json-like-entry"])
def test_a_hit_gets_its_own_list_over_the_same_rows(json_like):
    rows = _rows()
    srt = sorted(rows, key=lambda r: r[1])
    backend = InMemoryBackend()
    backend.set("k", _entry({"rows": rows, "srt": srt, "n": len(rows)}, json_like))
    assert type(backend._store["k"][1]) is not _Marshalled
    assert type(_stored_rows(backend, "k", "rows")) is list

    for _ in range(2):
        _meta, got = backend.get("k")
        assert got["variables"]["rows"] == rows
        assert got["variables"]["srt"] == srt
        assert got["variables"]["rows"] is not rows
        assert got["variables"]["rows"] is not _stored_rows(backend, "k", "rows")
        assert got["variables"]["rows"][0] is _stored_rows(backend, "k", "rows")[0]  # the row is shared
        got["variables"]["rows"].append(("x",))
        got["variables"]["srt"].clear()
    _meta, again = backend.get("k")
    assert again["variables"]["rows"] == rows
    assert again["variables"]["srt"] == srt


def test_two_names_for_one_list_still_share_it():
    rows = _rows()
    backend = InMemoryBackend()
    backend.set("k", _entry({"a": rows, "b": rows}))
    _meta, got = backend.get("k")
    assert got["variables"]["a"] is got["variables"]["b"]
    assert got["variables"]["a"] == rows


def test_the_callers_list_changed_later_does_not_reach_the_entry():
    rows = _rows()
    backend = InMemoryBackend()
    backend.set("k", _entry({"rows": rows}))
    rows.append(("late",))
    rows[0] = ("changed",)
    _meta, got = backend.get("k")
    assert got["variables"]["rows"] == _rows()


def test_rows_beside_a_closed_file_come_back_as_rows(tmp_path):
    path = tmp_path / "rows.txt"
    path.write_text("", encoding="utf-8")
    with open(path, encoding="utf-8") as fh:
        pass
    rows = _rows()
    backend = InMemoryBackend()
    backend.set("k", _entry({"rows": rows, "fh": stored_form(fh)}))
    _meta, got = backend.get("k")
    assert got["variables"]["rows"] == rows
    assert type(got["variables"]["rows"]) is list
    assert got["variables"]["fh"].closed
    _meta, peeked = backend.peek_entry("k")
    assert peeked["variables"]["rows"] == rows


def test_a_list_holding_lists_is_copied_as_before():
    nested = [[i, i + 1] for i in range(N)]
    backend = InMemoryBackend()
    backend.set("k", _entry({"nested": nested}))
    _meta, got = backend.get("k")
    assert got["variables"]["nested"] == nested
    got["variables"]["nested"][0].append(9)
    _meta, again = backend.get("k")
    assert again["variables"]["nested"][0] == [0, 1]
