"""A value measured against a nearly full disk is pickled once.

When the disk tier's cap (scaled to free disk) is below a value's size in
RAM, the tier measures what the value pickles to before refusing it: a
frame of strings pickles to a fraction of its footprint. The measure was a
whole pickle on the caller's thread, and the writer then pickled the value
again: 2.6 s per grid of a loop on a nearly full disk. The bytes measured
are now the bytes written.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cash.backends.file_backend import FileBackend
from cash.backends.memory_backend import InMemoryBackend
from cash.backends.serialization import PickleSerializer
from cash.backends.tiered_backend import TieredBackend
from tests._work_counts import method_calls

ROWS = 200_000


def _frame() -> pd.DataFrame:
    """Weighed at 10x what it pickles to: one string object in every row."""
    name = "player-" + "x" * 40
    return pd.DataFrame({"name": pd.Series([name] * ROWS, dtype=object), "x": np.arange(ROWS, dtype=float)})


def _tiers(tmp_path, cap: int) -> TieredBackend:
    return TieredBackend(
        [InMemoryBackend(), FileBackend(str(tmp_path / "cache"), max_size_bytes=cap, flush_interval=0)]
    )


def test_a_value_that_fits_once_pickled_is_pickled_once_and_written(tmp_path):
    df = _frame()
    tiers = _tiers(tmp_path, cap=8 * 1024**2)  # under its 20 MB in RAM, over its 2 MB pickled
    with (
        method_calls(PickleSerializer, "serialize") as whole,
        method_calls(PickleSerializer, "serialize_split") as split,
    ):
        tiers.set("k", df, {"execution_time": 5.0, "force_persist": True})
        tiers.backends[1]._writes.wait_all()
    assert whole.calls + split.calls == 1
    pd.testing.assert_frame_equal(FileBackend(str(tmp_path / "cache"), flush_interval=0).get("k")[1], df)


def test_a_value_over_the_cap_once_pickled_is_still_refused(tmp_path):
    df = _frame()
    tiers = _tiers(tmp_path, cap=64 * 1024)
    metadata = {"execution_time": 5.0, "force_persist": True}
    tiers.set("k", df, metadata)
    tiers.backends[1]._writes.wait_all()
    assert FileBackend(str(tmp_path / "cache"), flush_interval=0).get("k")[0] is None
    assert metadata["persist_skipped"] == "size"


def test_a_caller_change_after_the_store_does_not_reach_the_disk(tmp_path):
    """No RAM tier to give the writer a copy of its own: the bytes measured
    are copied, as the writer's own would be."""
    df = _frame()
    first = FileBackend(str(tmp_path / "first"), flush_interval=0)
    tiers = TieredBackend([first, FileBackend(str(tmp_path / "cache"), max_size_bytes=8 * 1024**2, flush_interval=0)])
    tiers.set("k", df, {"execution_time": 5.0, "force_persist": True, "size": 20 * 1024**2})
    df.loc[0, "x"] = -1.0  # nothing else shares the block: written in place
    tiers.backends[1]._writes.wait_all()
    got = FileBackend(str(tmp_path / "cache"), flush_interval=0).get("k")[1]
    assert got["x"].iloc[0] == 0.0
