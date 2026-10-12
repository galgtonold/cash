"""A table from the RAM tier keeps the block layout the caller's table had.

pandas joins the columns of one dtype into one block on a deep copy, and the
tier froze a joined table. ``data[["f1", "f2"]].to_numpy()`` on one block is
a view, read-only under copy-on-write, where the caller's own table (one
block per column, as ``read_csv`` builds it) gives a new, writable array: the
user's ``X -= X.mean(axis=0)`` raised "output array is read-only" on a
restored table only.
"""

from __future__ import annotations

import io
import pickle

import numpy as np
import pandas as pd
import pytest

from cash.backends import frame_sharing
from cash.backends.memory_backend import InMemoryBackend

CSV = pd.DataFrame({"f1": np.arange(50.0), "f2": np.arange(50.0) ** 0.5, "f3": np.ones(50)}).to_csv(index=False)


def _read() -> pd.DataFrame:
    return pd.read_csv(io.StringIO(CSV))


def _hit_after_store() -> pd.DataFrame:
    backend = InMemoryBackend()
    backend.set("k", _read(), {"execution_time": 1.0})
    return backend.get("k")[1]


def _hit_after_disk() -> pd.DataFrame:
    backend = InMemoryBackend()
    assert backend.promote("k", pickle.loads(pickle.dumps(_read())), {"execution_time": 1.0})
    return backend.get("k")[1]


@pytest.mark.parametrize(
    "hit",
    [
        _hit_after_store,
        pytest.param(
            _hit_after_disk,
            marks=pytest.mark.skipif(not frame_sharing.enabled(), reason="a table read from disk is kept frozen"),
        ),
    ],
    ids=["stored", "read_from_disk"],
)
def test_to_numpy_of_a_hit_is_as_writable_as_of_the_callers_table(hit):
    plain = _read()
    if frame_sharing.enabled():
        assert plain._mgr.nblocks == 3, "setup: pandas 3's read_csv builds one block per column"
    expected = plain[["f1", "f2"]].to_numpy().flags.writeable
    table = hit()
    assert table._mgr.nblocks == plain._mgr.nblocks
    X = table[["f1", "f2"]].to_numpy()
    assert X.flags.writeable == expected
    if X.flags.writeable:
        X -= X.mean(axis=0)  # the user's own array: writing it touches no entry
        assert hit()["f1"].iloc[0] == 0.0


@pytest.mark.skipif(not frame_sharing.enabled(), reason="needs pandas copy-on-write")
def test_a_hit_of_an_unjoined_table_is_still_frozen():
    """Each block is frozen on its own: no handle writes the entry."""
    backend = InMemoryBackend()
    backend.set("k", _read(), {"execution_time": 1.0})
    hit = backend.get("k")[1]
    with pytest.raises(ValueError):
        hit["f2"].array[0] = 99.0
    hit.iloc[0, 1] = 99.0  # through pandas: copies first
    assert backend.get("k")[1]["f2"].iloc[0] == 0.0
