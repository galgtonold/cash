"""An entry goes past RAM only when one re-run saves more than writing it costs.

What a re-run saves is the compute less reading the entry back; what the
write costs is its bytes at the rate this machine's recent writes of that
kind of value ran at (`write_cost.WriteRates`), the fitted cost model's
prediction until one was measured. ``@cash.cache``, ``@cash:persist`` and a
caller's own promotion rule decide for themselves.
"""

from __future__ import annotations

import pytest

from cash import cost_model
from cash.backends import FileBackend, InMemoryBackend, TieredBackend, write_cost
from cash.backends.persistence_policy import EXPECTED_RERUNS, PersistencePolicy, write_seconds
from cash.backends.sampled_cost import SampledCost

np = pytest.importorskip("numpy")

MiB = 1 << 20


def _array_entry(seconds: float, nbytes: int = 20 * MiB, **extra) -> dict:
    return {
        "execution_time": seconds,
        "size": nbytes,
        "cost_model_size_bytes": nbytes,
        "cost_model_type_name": "ndarray",
        "cost_model_family": "ndarray_dense",
        **extra,
    }


# -- the rule ---------------------------------------------------------------


def test_one_rerun_must_repay_the_write():
    assert EXPECTED_RERUNS == 1
    assert PersistencePolicy.pays_for_its_write(compute_s=2.0, restore_s=0.5, write_s=1.0)
    assert not PersistencePolicy.pays_for_its_write(compute_s=2.0, restore_s=0.5, write_s=1.5)
    assert not PersistencePolicy.pays_for_its_write(compute_s=1.4, restore_s=0.3, write_s=2.0)  # 22 MB at 11 MB/s
    assert PersistencePolicy.pays_for_its_write(compute_s=0.0, restore_s=0.0, write_s=0.0)  # nothing to repay


def test_a_write_dearer_than_the_rerun_saves_stays_in_ram():
    policy = PersistencePolicy()
    entry = _array_entry(1.0)
    kept = policy.decide("k", entry, backend_kind="disk", write_s=0.2)
    refused = policy.decide("k", entry, backend_kind="disk", write_s=2.0)
    assert kept.persist
    assert not refused.persist and refused.skipped == "write"


def test_a_slow_small_entry_is_written_on_a_slow_disk():
    # 1 KB computed in 60 s: the write is nothing next to the compute.
    policy = PersistencePolicy()
    entry = _array_entry(60.0, nbytes=1024)
    assert policy.decide("k", entry, backend_kind="disk", write_s=1024 / (1 * MiB)).persist


@pytest.mark.parametrize("chosen", [{"force_persist": True}, {"decorator_entry": True}])
def test_an_explicit_choice_is_never_refused_for_its_write(chosen):
    policy = PersistencePolicy()
    entry = _array_entry(0.0, **chosen)
    assert policy.decide("k", entry, backend_kind="disk", write_s=1e6).persist


def test_a_callers_promotion_rule_decides_alone():
    policy = PersistencePolicy()
    entry = {"execution_time": 0.001, "size": 1}
    assert policy.decide("k", entry, backend_kind="disk", override=lambda t, s: True, write_s=1e6).persist


def test_the_end_of_cell_write_must_repay_itself_too():
    policy = PersistencePolicy()
    entry = _array_entry(0.2)
    assert policy.decide_rebuild(entry, 3.0, backend_kind="disk", write_s=1.0).persist
    refused = policy.decide_rebuild(entry, 3.0, backend_kind="disk", write_s=5.0)
    assert not refused.persist and refused.skipped == "write"
    assert policy.decide_rebuild({**entry, "force_persist": True}, 3.0, backend_kind="disk", write_s=5.0).persist


# -- what a write costs -------------------------------------------------------


def test_until_measured_the_cost_model_prices_the_write():
    entry = _array_entry(1.0)
    rates = write_cost.WriteRates()
    assert write_seconds(entry, 20 * MiB, [rates]) == pytest.approx(
        cost_model.estimated_serialize_time("ndarray", 20 * MiB, "disk")
    )


def test_a_measured_rate_prices_the_write_by_kind():
    rates = write_cost.WriteRates()
    rates.record("ndarray_dense", 10 * MiB, 1.0)  # 10 MiB/s
    assert write_seconds(_array_entry(1.0), 20 * MiB, [rates]) == pytest.approx(2.0)
    # A frame is another kind, still priced by the model.
    frame = {**_array_entry(1.0), "cost_model_type_name": "DataFrame", "cost_model_family": "dataframe_numeric"}
    assert write_seconds(frame, 20 * MiB, [rates]) < 0.1


def test_the_rate_follows_recent_writes_and_ignores_small_ones():
    rates = write_cost.WriteRates()
    rates.record("k", 1000, 10.0)  # under a MiB: its fixed costs, not a rate
    assert rates.per_byte("k") is None
    rates.record("k", 10 * MiB, 1.0)
    rates.record("k", 10 * MiB, 3.0)
    assert rates.per_byte("k") == pytest.approx(2.0 / (10 * MiB))  # halfway to the newest
    assert rates.seconds("k", 100, assumed=0.5) == 0.5  # a small write: its assumed cost


def test_many_python_objects_are_their_own_kind_and_never_cheaper_than_their_sample():
    rates = write_cost.WriteRates()
    rates.record("dataframe_numeric", 10 * MiB, 0.01)
    frame = {**_array_entry(1.0), "cost_model_type_name": "DataFrame", "cost_model_family": "dataframe_numeric"}
    sampled = SampledCost(dump=1.5, load=0.2)
    assert write_seconds(frame, 20 * MiB, [rates], sampled) == pytest.approx(1.5)
    rates.record(write_cost.OBJECTS_KIND, 10 * MiB, 10.0)
    assert write_seconds(frame, 20 * MiB, [rates], sampled) == pytest.approx(20.0)


def test_each_tier_pays_its_own_write():
    one, two = write_cost.WriteRates(), write_cost.WriteRates()
    for rates in (one, two):
        rates.record("ndarray_dense", 10 * MiB, 1.0)
    assert write_seconds(_array_entry(1.0), 20 * MiB, [one, two]) == pytest.approx(4.0)
    assert write_seconds(_array_entry(1.0), 20 * MiB, []) == 0.0


def test_the_call_entries_it_refers_to_are_written_with_it():
    # ``h = run(data, k)``: a few bytes of its own, the call's frame by reference.
    rates = write_cost.WriteRates()
    rates.record(write_cost.OBJECTS_KIND, 10 * MiB, 1.0)  # 10 MiB/s, the slowest kind measured
    rates.record("ndarray_dense", 10 * MiB, 0.01)
    statement = {"execution_time": 1.0, "size": 300, "call_ref_bytes": 20 * MiB}
    assert write_seconds(statement, 300, [rates]) == pytest.approx(2.0, rel=0.01)
    assert not PersistencePolicy().decide("k", statement, backend_kind="disk", write_s=2.0).persist


# -- through the tiered backend -------------------------------------------------


@pytest.fixture
def tiers(tmp_path, monkeypatch):
    # Measured, as in use: the suite pins every queue's rate otherwise.
    monkeypatch.delenv(write_cost.PIN_ENV, raising=False)
    disk = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    backend = TieredBackend([InMemoryBackend(), disk])
    yield backend, disk
    disk._writes.wait_all()
    disk.shutdown()


def _set(backend, key, seconds):
    value = {"variables": {"a": np.arange(20 * MiB // 8, dtype="float64")}}
    meta = _array_entry(seconds)
    backend.set(key, value, meta)
    return meta


def test_a_write_teaches_its_tier_what_writes_of_that_kind_cost(tiers):
    backend, disk = tiers
    assert disk._writes.rates.per_byte("ndarray_dense") is None
    meta = _set(backend, "first", 5.0)
    disk._writes.wait_all()
    assert "DISK" in meta["storage"]
    assert disk._writes.rates.per_byte("ndarray_dense") is not None


def test_a_disk_measured_slow_keeps_a_quick_large_entry_in_ram(tiers):
    backend, disk = tiers
    disk._writes.rates.record("ndarray_dense", 100 * MiB, 10.0)  # 10 MiB/s: 2 s for this entry
    slow = _set(backend, "quick", 1.0)
    assert slow["storage"] == ["RAM"] and slow["persist_skipped"] == "write"
    # The same entry from a long compute repays its write.
    long = _set(backend, "long", 5.0)
    assert "DISK" in long["storage"]


def test_a_pinned_rate_is_not_moved_by_the_writes(tmp_path, monkeypatch):
    # What the suite sets so its store decisions do not depend on the load.
    monkeypatch.setenv(write_cost.PIN_ENV, "10")
    disk = FileBackend(str(tmp_path / "cache"), flush_interval=0)
    backend = TieredBackend([InMemoryBackend(), disk])
    try:
        meta = _set(backend, "long", 5.0)  # 20 MiB at 10 MiB/s: 2 s, repaid by 5 s
        disk._writes.wait_all()
        assert "DISK" in meta["storage"]
        assert disk._writes.rates.per_byte("ndarray_dense") == pytest.approx(1 / (10 * MiB))
        quick = _set(backend, "quick", 1.0)
        assert quick["persist_skipped"] == "write"
    finally:
        disk._writes.wait_all()
        disk.shutdown()
