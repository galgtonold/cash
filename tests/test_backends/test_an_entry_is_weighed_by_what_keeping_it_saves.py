"""An entry is weighed by what keeping it saves (``store_time``), when its
writer knows that is less than what it cost (``execution_time``).

A notebook statement around a cached call costs the call's time too, and a
hit is credited with all of it. But the call's own entry holds that: keeping
the statement's value saves only the rest. The disk decision and the eviction
ranking both weigh that, so a statement is not written to disk, or kept over
another entry, for work its call's entry already saves.
"""

from __future__ import annotations

from cash.backends import FileBackend, InMemoryBackend, TieredBackend
from cash.backends._base import gdsf_value, store_seconds
from cash.backends.persistence_policy import PersistencePolicy

MIB = 1024**2


def _meta(**extra):
    return {"size": 2 * MIB, "cost_model_family": "dataframe", "cost_model_type_name": "DataFrame", **extra}


def test_the_disk_decision_weighs_the_store_time():
    policy = PersistencePolicy()
    assert policy.decide("k", _meta(execution_time=5.0), backend_kind="disk").persist
    held = policy.decide("k", _meta(execution_time=5.0, store_time=0.02), backend_kind="disk")
    assert not held.persist and held.skipped == "compute"


def test_the_eviction_ranking_weighs_the_store_time():
    full = gdsf_value({"execution_time": 5.0}, MIB)
    own = gdsf_value({"execution_time": 5.0, "store_time": 0.05}, MIB)
    assert own < full / 50


def test_without_a_store_time_the_execution_time_is_what_it_saves():
    assert store_seconds({"execution_time": 1.5}) == 1.5
    assert store_seconds({"execution_time": 1.5, "store_time": 0.0}) == 0.0
    assert store_seconds({}) == 0.0


def test_through_the_tiered_backend(tmp_path):
    backend = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path), flush_interval=0)])
    kept = _meta(execution_time=5.0, store_time=4.0)
    held = _meta(execution_time=5.0, store_time=0.02)
    backend.set("stmt:kept", {"variables": {"v": b"x" * MIB}}, kept)
    backend.set("stmt:held", {"variables": {"v": b"x" * MIB}}, held)
    backend.shutdown()
    assert "DISK" in kept["storage"], kept
    assert kept["storage"] and held["storage"] == ["RAM"], held
