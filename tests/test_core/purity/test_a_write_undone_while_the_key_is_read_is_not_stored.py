"""A list written while its key is read, and put back before the body, is
not stored under the key of the written items.

Another thread swapped ``rows[5]`` as the key was read and swapped it back
before the body: the key held the written item, the body read the original,
and the identities the check after the body compares, taken as the body
started, had not moved, so the plain list was not hashed again. The result
was stored, and a later call with the written item was served the
original's. A list of leaves is now keyed from the copy of its items its
census took and vouched for only while it still holds them; any other
list is hashed again after the body (`arg_hashing.KeyedLists`).

The other thread is played here by the call's own steps: the write at the
argument's key part, the undo where the identities are taken.
"""

from __future__ import annotations

import pytest

from cash import Cash
from cash.decorator import arg_hashing, purity_checks

pytestmark = pytest.mark.filterwarnings("ignore::cash.exceptions.CashWarning")


def _ints(n):
    def put(rows, i, item):
        rows[i] = item

    return list(range(n)), (lambda rows, i: rows[i]), put, 10**6, sum


def _rows(n):
    def put(rows, i, item):
        rows[i][0] = item

    return [[i, i] for i in range(n)], (lambda rows, i: rows[i][0]), put, 10**6, lambda rows: sum(a for a, _b in rows)


def _tuples(n):
    def put(rows, i, item):
        rows[i] = item

    return [(i, i) for i in range(n)], (lambda rows, i: rows[i]), put, (10**6, 5), lambda rows: sum(a for a, _b in rows)


SHAPES = {
    "500 ints": lambda: _ints(500),
    "5000 ints": lambda: _ints(5000),
    "rows of lists": lambda: _rows(100),
    "3000 rows of lists": lambda: _rows(3000),
    "rows of tuples": lambda: _tuples(3000),
}


@pytest.mark.parametrize("backend", ["disk", "memory"])
@pytest.mark.parametrize("shape", list(SHAPES.values()), ids=list(SHAPES))
def test_the_written_items_are_not_served_the_originals_result(tmp_path, monkeypatch, backend, shape):
    rows, get, put, written, total = shape()
    cash = (
        Cash(backend="memory", register_magic=False)
        if backend == "memory"
        else Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    )
    f = cash.cache(lambda rows: total(rows))
    for _ in range(3):
        f(rows)
    put(rows, 7, get(rows, 8))  # a change, so the next call misses
    original = get(rows, 5)
    key_part, identities = arg_hashing.plain_key_part, purity_checks.PurityChecks.argument_identities
    state: dict = {}

    def write_while_keyed(value):
        if value is rows and "written" not in state:
            state["written"] = True
            put(rows, 5, written)
        return key_part(value)

    def undo_before_body(self, *args, **kwargs):
        if state.get("written") and "undone" not in state:
            state["undone"] = True
            put(rows, 5, original)
        return identities(self, *args, **kwargs)

    monkeypatch.setattr(arg_hashing, "plain_key_part", write_while_keyed)
    monkeypatch.setattr(purity_checks.PurityChecks, "argument_identities", undo_before_body)
    assert f(rows) == total(rows)
    monkeypatch.undo()
    assert state == {"written": True, "undone": True}
    put(rows, 5, written)
    assert f(rows) == total(rows)
