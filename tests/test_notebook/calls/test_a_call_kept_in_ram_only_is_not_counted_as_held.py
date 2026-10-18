"""A call whose entry stays in RAM only is not counted as held past a restart.

A statement's own work is its cost less the calls the cache holds
(``CallUnit.cached_compute_s``): after a restart, those calls are restored
rather than run. A call above the persistence floor whose result was too big
for its compute on disk is kept in RAM only, and runs again after a restart,
so it is part of the statement's work. Counted as held, the statement was
priced under what a restart costs, for its own disk decision and for its
rank in eviction.
"""

from __future__ import annotations

import time

import pytest

from cash.notebook.call_interception import CallSite
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def _unit_whose_backend_reports(call_unit_harness, skipped):
    def slow(v):
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return [v]

    unit = call_unit_harness(lineage={"x": "h"}, user_ns={"x": 1, "slow": slow})
    backend = unit._cash.backend
    real_set = backend.set

    def reporting_set(key, value, metadata=None, *args, **kwargs):
        result = real_set(key, value, metadata, *args, **kwargs)
        # What `TieredBackend.set` notes when it keeps an entry from disk.
        if metadata is not None and skipped is not None:
            metadata["persist_skipped"] = skipped
        return result

    backend.set = reporting_set
    call = unit.wrap(slow, CallSite(source="slow(x)", free_names=frozenset({"slow", "x"}), occurrence_index=0))
    return unit, call


@pytest.mark.parametrize(
    ("skipped", "held"),
    [
        pytest.param("bytes", False, id="too big for its compute on disk"),
        pytest.param("size", False, id="over every disk tier's size cap"),
        pytest.param(None, True, id="written past RAM, or no tier past RAM"),
    ],
)
def test_only_a_call_kept_past_a_restart_counts(call_unit_harness, skipped, held):
    unit, call = _unit_whose_backend_reports(call_unit_harness, skipped)
    assert call(1) == [1]
    assert [e["stored"] for e in unit.drain()] == [True], "the call must be stored for this to test anything"
    if held:
        assert unit.cached_compute_s >= ABOVE_PERSISTENCE_FLOOR_S * 0.9
    else:
        assert unit.cached_compute_s == 0.0
        assert unit.cached_restore_s == 0.0
