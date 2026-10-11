"""A big value of many Python objects whose write would cost more than a
restore saves stays in the RAM tier.

The disk write runs in the background, but pickling holds the interpreter
lock, so the notebook waits for it. A dict of 87,000 users' parsed sessions,
computed in 3.0 s, took 3.7 s to pickle and restored in 1.25 s: the
statement took 10.2 s under cash. The write and the restore are now
measured on a sample (`sampled_cost`) instead of priced by the value's bytes.
"""

from __future__ import annotations

import datetime

from cash.backends.persistence_policy import PersistencePolicy
from cash.backends.sampled_cost import SampledCost, sampled_cost


def _sessions(users: int) -> dict:
    day = datetime.datetime(2023, 1, 1)
    return {f"User{u}": [[("Action_1", day + datetime.timedelta(seconds=j)) for j in range(5)]] for u in range(users)}


def test_a_big_object_heavy_value_is_measured_on_a_sample():
    cost = sampled_cost({"variables": {"data": _sessions(40_000)}})

    assert cost is not None and cost.dump > 0 and cost.load > 0


def test_small_or_numeric_values_are_left_to_the_byte_model():
    assert sampled_cost({"variables": {"data": _sessions(100)}}) is None
    assert sampled_cost(list(range(100))) is None


def _decide(compute_s: float, sampled: SampledCost | None):
    metadata = {"size": 300 << 20, "execution_time": compute_s}
    return PersistencePolicy().decide("k", metadata, backend_kind="disk", sampled=sampled)


def test_a_write_longer_than_what_a_restore_saves_is_not_persisted():
    assert not _decide(3.0, SampledCost(dump=3.7, load=1.25)).persist


def test_a_write_that_pays_back_on_one_restore_is_persisted():
    assert _decide(30.0, SampledCost(dump=3.7, load=1.25)).persist


def test_without_a_sample_the_byte_model_decides_as_before():
    assert _decide(3.0, None).persist
