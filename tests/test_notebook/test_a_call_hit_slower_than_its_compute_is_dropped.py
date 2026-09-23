"""A call-cache hit that costs more than the compute it saves is dropped.

`net_returns(orders, w)` hits took ~10 s each against ~4 s
of compute, and the badge reported "4/4 hit". The call cache had no
restore-cost check at all -- the statement path has one -- and nothing
compared a hit's real cost with what it saved. Measured on the hit now:
a loss drops the entry and the site runs plain for the rest of the session.
"""

import time

from cash.notebook.call_interception import CallSite


def _unit(call_unit_harness, calls):
    def work(k):
        calls.append(k)
        time.sleep(0.02)
        return k * 2

    unit = call_unit_harness(lineage={"k": "hash-2"}, user_ns={"k": 2, "work": work})
    site = CallSite(source="work(k)", free_names=frozenset({"work", "k"}), occurrence_index=0)
    return unit, unit.wrap(work, site)


def test_a_hit_slower_than_its_compute_is_dropped(call_unit_harness, monkeypatch):
    calls: list[int] = []
    unit, wrapped = _unit(call_unit_harness, calls)
    assert wrapped(2) == 4 and calls == [2]
    backend = unit._cash.backend
    real_get = backend.get

    def slow_get(key, *a, **k):
        time.sleep(0.3)  # a hit that costs far more than 20 ms
        return real_get(key, *a, **k)

    monkeypatch.setattr(backend, "get", slow_get)
    assert wrapped(2) == 4
    assert calls == [2], "the second call should still have been a hit"
    assert wrapped(2) == 4
    assert calls == [2, 2], "a hit that cost 0.3 s to save 0.02 s was served again"


def test_a_hit_that_pays_is_kept(call_unit_harness):
    calls: list[int] = []
    _unit_, wrapped = _unit(call_unit_harness, calls)
    for _ in range(3):
        assert wrapped(2) == 4
    assert calls == [2]
