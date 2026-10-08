"""Big module data is hashed for an outside change only where it is read.

Once ``v = helpers.score(3)`` had read a 256 MB table, every cell using ``v``
(``a = v + 1``) hashed the whole table first to look for a change made outside
the notebook: 250 ms a cell. Before a cell that only derives from the reader
(`recorded_reads.REBOUND`), a big value still bound to the object last hashed is not hashed
again; one rebound since is, and so is a small one. A change made to a big one
in place is seen by the next cell that reads the data itself, and inside the
upstream simulation each value is hashed once (`one_reading`).
"""

from __future__ import annotations

import sys
import types

import pytest

from cash.notebook import recorded_reads
from cash.notebook.recorded_reads import ReadRecord, outside_changes

np = pytest.importorskip("numpy")

BIG = 1_000_000  # 8 MB of float64: above the bar
NAME = "statelib_big_data_probe"


@pytest.fixture
def statelib():
    module = types.ModuleType(NAME)
    module.__file__ = f"/tmp/{NAME}.py"
    module.TABLE = np.zeros(BIG)
    module.SMALL = list(range(1000))
    sys.modules[NAME] = module
    yield module
    del sys.modules[NAME]


@pytest.fixture
def hashed(monkeypatch):
    labels: list[str] = []
    real = recorded_reads.module_data_digest

    def counting(label, value):
        labels.append(label)
        return real(label, value)

    monkeypatch.setattr(recorded_reads, "module_data_digest", counting)
    return labels


def _record(*attrs):
    record = ReadRecord()
    record.watched.update(("mod", f"{NAME}.{attr}") for attr in attrs)
    outside_changes(record)  # the first look takes them as known
    return record


def test_a_big_value_still_the_same_object_is_not_hashed(statelib, hashed):
    record = _record("TABLE")
    hashed.clear()

    statelib.TABLE[0] = 5.0  # in place, from outside the notebook
    assert outside_changes(record, module_data=recorded_reads.REBOUND) is False
    assert hashed == []
    # A cell that reads the data still sees the change.
    assert outside_changes(record, module_data=True) is True
    assert hashed == [f"{NAME}.TABLE"]


def test_a_big_value_rebound_is_hashed(statelib, hashed):
    record = _record("TABLE")
    hashed.clear()

    statelib.TABLE = np.ones(BIG)
    assert outside_changes(record, module_data=recorded_reads.REBOUND) is True
    assert hashed == [f"{NAME}.TABLE"]
    # The new object is the one known now.
    hashed.clear()
    assert outside_changes(record, module_data=recorded_reads.REBOUND) is False
    assert hashed == []


def test_a_small_value_is_still_hashed_for_an_in_place_change(statelib, hashed):
    record = _record("SMALL")
    hashed.clear()

    statelib.SMALL[0] = -1
    assert outside_changes(record, module_data=recorded_reads.REBOUND) is True
    assert hashed == [f"{NAME}.SMALL"]


def test_inside_one_reading_each_value_is_hashed_once(statelib, hashed):
    label = f"{NAME}.TABLE"
    with recorded_reads.one_reading():
        first = recorded_reads._current("mod", label)
        assert recorded_reads._current("mod", label) == first
    assert hashed == [label]
    recorded_reads._current("mod", label)
    assert hashed == [label, label], "outside the block every read hashes"


def test_known_digests_stand_for_what_the_look_did_not_hash(statelib, hashed):
    record = _record("TABLE", "SMALL")
    known = dict(record.known)
    hashed.clear()
    statelib.TABLE[0] = 5.0
    with recorded_reads.one_reading():
        recorded_reads.take_known_as_read(record)
        assert recorded_reads._current("mod", f"{NAME}.TABLE") == known[("mod", f"{NAME}.TABLE")]
    assert hashed == []
    # Not after the block, nor for an object rebound since.
    assert recorded_reads._current("mod", f"{NAME}.TABLE") != known[("mod", f"{NAME}.TABLE")]
    statelib.SMALL = [1, 2]
    with recorded_reads.one_reading():
        recorded_reads.take_known_as_read(record)
        assert recorded_reads._current("mod", f"{NAME}.SMALL") != known[("mod", f"{NAME}.SMALL")]
