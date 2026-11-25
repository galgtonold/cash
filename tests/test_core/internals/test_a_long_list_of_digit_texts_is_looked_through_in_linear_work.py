"""A result that is a long list of texts full of digits is looked through
for a live object's ``id()`` in work that grows with the list.

A run of more than fifteen digits can spell an address inside it
(``f"{id(a)}{id(b)}"``), so each such run was cut into every part that
could be one, in Python, and the look read the list twice: 300,000 texts
of 18 digits made a million numbers and took 5 s where the function took
40 ms. The parts are now searched for the leading digits an address of
this process has, and the second read finds the first.
"""

from __future__ import annotations

import sys
import warnings
from itertools import chain

import pytest

from cash.notebook.statement import identity_holders
from tests._work_counts import method_calls

linux = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the process's mappings are read on Linux")

ROWS = 20_000


def _digit_texts(rows=ROWS):
    return [str(100_000 + i) * 3 for i in range(rows)]


class _Thing:
    def __init__(self):
        self.name = "thing"


KEEP = _Thing()


def _codes(caught):
    return [getattr(w.message, "code", None) for w in caught]


@linux
def test_the_search_finds_every_part_the_run_by_run_read_keeps(monkeypatch):
    a, b = _Thing(), _Thing()
    # One reading of the mappings for both: the heap grows while the test
    # builds its sets, and a part inside the new piece is no object's id.
    ranges = identity_holders._read_mappings()
    monkeypatch.setattr(identity_holders, "_read_mappings", lambda: ranges)
    monkeypatch.setattr(identity_holders, "_MAPPINGS", [None])
    runs = _digit_texts(2_000)
    runs += [f"{id(a)}{id(b)}", f"77{id(a)}", f"{id(b)}1234", f"{id(a)}" * 3, "0" * 20, "9" * 40]
    runs += [f"{i}{id(a) + 16 * i}{i}" for i in range(50)]
    assert len(runs) >= identity_holders._MANY_RUNS
    one_by_one = set(chain.from_iterable(map(identity_holders._windows, runs)))
    searched = identity_holders._mapped_windows(runs)
    assert searched is not None
    kept = identity_holders._candidate_ids(one_by_one)
    assert {id(a), id(b)} <= kept, "the control: the run-by-run read finds the ids"
    assert identity_holders._candidate_ids(searched) == kept
    # The search makes few numbers that are no address.
    assert len(searched) < len(one_by_one) / 4


@linux
def test_a_long_list_of_digit_texts_is_not_cut_up_run_by_run(disk_cash):
    runs = []

    @disk_cash.cache
    def make(n):
        runs.append(n)
        return _digit_texts(n)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with method_calls(identity_holders, "_windows") as cut:
            first = make(ROWS)
        second = make(ROWS)
    assert first == second == _digit_texts()
    assert runs == [ROWS], "the list was not stored"
    assert cut.calls == 0, f"{cut.calls} runs of digits were cut up one at a time"
    assert not {"STORE-IDENTITY-HELD", "CACHE-IDENTITY-UNCHECKED"} & set(_codes(caught))


def test_the_list_is_read_once_when_a_number_in_it_could_be_an_id(disk_cash):
    # The address of an object nothing the function can reach holds: a
    # number that could be an id(), so the quick look at plain data leaves
    # the answer to the walk, which read the list a second time.
    stranger = _Thing()
    spelled = str(id(stranger))
    runs = []

    @disk_cash.cache
    def make(n):
        runs.append(n)
        return [*_digit_texts(n), spelled]

    with method_calls(identity_holders, "_spelled_in") as read:
        make(ROWS)
    make(ROWS)
    assert runs == [ROWS], "an address of an object the function cannot reach kept the list out of the cache"
    assert read.calls == 1, f"the list's text was read {read.calls} times"


def test_two_ids_glued_together_among_many_digit_texts_are_still_found(disk_cash):
    runs = []

    @disk_cash.cache
    def make(n):
        runs.append(n)
        return [*_digit_texts(n), f"{id(KEEP)}{id(KEEP)}"]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make(ROWS)
        make(ROWS)
    assert runs == [ROWS, ROWS], "a result spelling a global's id() was served from the cache"
    assert "STORE-IDENTITY-HELD" in _codes(caught)
