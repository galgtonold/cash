"""A list of many small pandas tables is priced by a sample of its tables.

Each small table costs a fixed fraction of a millisecond to copy into the
RAM tier and to pickle, whatever its bytes: a loop building 6,000
three-row frames in 1.2 s cost 1.1 s to copy and 1.15 s to pickle, which
the byte model read as free. `many_frames_cost` measures a sample of the
tables and scales by their count; the notebook keeps such a value as
metadata only when storing it costs more than computing it
(`PersistencePolicy.refuses_store`).
"""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from cash.backends.persistence_policy import PersistencePolicy
from cash.backends.sampled_cost import many_frames_cost, sampled_cost


def _frames(count: int, rows: int = 3) -> list:
    return [pd.DataFrame({"user": ["s10"] * rows, "t": range(i, i + rows)}) for i in range(count)]


def test_many_small_tables_are_priced_by_their_count():
    cost = many_frames_cost({"df_array": _frames(1_000)})

    assert cost is not None
    assert cost.items == 1_000
    assert cost.copy > 0 and cost.dump > 0 and cost.load > 0


def test_the_write_decision_sees_them_too():
    """The disk tier's decision (`sampled_cost`) prices the same list by
    its tables, without the RAM copy it has made already."""
    cost = sampled_cost({"variables": {"df_array": _frames(1_000)}})

    assert cost is not None and cost.dump > 0 and cost.copy == 0


def test_a_few_tables_or_big_ones_are_left_to_the_byte_model():
    assert many_frames_cost({"df_array": _frames(100)}) is None
    assert many_frames_cost({"df_array": _frames(600, rows=10_000)}) is None
    assert many_frames_cost({"df_array": list(range(5_000))}) is None
    assert many_frames_cost({"frame": _frames(1, rows=50_000)[0]}) is None


def test_a_mixed_list_is_left_to_the_byte_model():
    mixed = _frames(600)
    mixed[::2] = [{"not": "a table"}] * len(mixed[::2])

    assert many_frames_cost({"df_array": mixed}) is None


def test_a_store_dearer_than_the_compute_is_refused():
    policy = PersistencePolicy()

    assert policy.refuses_store(1.2, 2.25)
    assert not policy.refuses_store(3.0, 2.25)
    assert not policy.refuses_store(0.0, 2.25)
