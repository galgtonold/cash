"""A list of many small pandas tables is priced by a sample of its tables.

Each small table costs a fixed fraction of a millisecond to copy into the
RAM tier and to pickle, whatever its bytes: a loop building 6,000
three-row frames in 1.2 s cost 1.1 s to copy and 1.15 s to pickle, which
the byte model read as free. `many_frames_cost` measures a sample of the
tables and scales by their count; the notebook keeps such a value as
metadata only when storing it costs more than a hit of it saves
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

    assert policy.refuses_store(1.2, copy_s=1.1, dump_s=1.15, restore_s=0.0)
    assert not policy.refuses_store(5.0, copy_s=1.1, dump_s=1.15, restore_s=0.0)
    assert not policy.refuses_store(0.0, copy_s=1.1, dump_s=1.15, restore_s=0.0)


def test_the_pickling_counts_twice_and_the_restore_against_the_saving():
    """Copying (0.55 s) and pickling (0.36 s) cost less than the compute
    (1.38 s), but the pickling holds the interpreter lock while the next
    statement runs, which waits as long again, and a hit from RAM copies
    every table again: 0.55 + 2 * 0.36 > 1.38 - 0.55, so it is refused."""
    policy = PersistencePolicy()

    assert policy.refuses_store(1.38, copy_s=0.55, dump_s=0.36, restore_s=0.55)


def test_the_pickling_alone_can_decide():
    """0.3 + 0.4 is under the 1.0 s compute; with the next statement's wait
    for the lock, 0.3 + 2 * 0.4 is over it."""
    policy = PersistencePolicy()

    assert policy.refuses_store(1.0, copy_s=0.3, dump_s=0.4, restore_s=0.0)
    assert not policy.refuses_store(1.0, copy_s=0.3, dump_s=0.3, restore_s=0.0)


def test_the_restore_alone_can_decide():
    """A store of 0.5 s against a 1.0 s compute is worth it unless a hit
    gives back less than that: restoring in 0.6 s, it saves 0.4 s."""
    policy = PersistencePolicy()

    assert policy.refuses_store(1.0, copy_s=0.3, dump_s=0.1, restore_s=0.6)
    assert not policy.refuses_store(1.0, copy_s=0.3, dump_s=0.1, restore_s=0.4)
