"""A table served from the cache holds its data in the memory order it had.

``pivot`` builds a frame whose one float block is F-ordered, so its
``.values`` is row-major (C). pandas' deep copy copies a block in C order, and
the RAM tier kept that copy: the restored frame's ``.values`` came back
column-major. Row-wise numpy (``held @ close.values[i]``) then summed in
another order, and a backtest differed from plain Python in the last bit
(1.4e-17). Values downstream of a hit must be bit-identical to a recompute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cash.backends.memory_backend import InMemoryBackend


def _pivoted() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    long = pd.DataFrame(
        {
            "date": np.repeat(np.arange(40), 30),
            "ticker": np.tile([f"T{i}" for i in range(30)], 40),
            "close": rng.random(1200),
        }
    )
    return long.pivot(index="date", columns="ticker", values="close")


def _f_ordered() -> pd.DataFrame:
    return pd.DataFrame(np.asfortranarray(np.random.default_rng(1).random((40, 30))))


def _with_dates() -> pd.DataFrame:
    frame = _pivoted()
    frame["a"] = pd.Timestamp("2020-01-01")
    frame["b"] = pd.Timestamp("2021-01-01")
    return frame


def _layout(frame: pd.DataFrame) -> list[tuple[bool, bool]]:
    """Each block's array's (C, F) contiguity; a datetime block by its ndarray."""
    arrays = [getattr(block.values, "_ndarray", block.values) for block in frame._mgr.blocks]
    return [(a.flags.c_contiguous, a.flags.f_contiguous) for a in arrays if isinstance(a, np.ndarray)]


@pytest.mark.parametrize("build", [_pivoted, _f_ordered, _with_dates], ids=["pivot", "fortran", "dates"])
def test_a_ram_hit_keeps_each_blocks_memory_order(build):
    frame = build()
    backend = InMemoryBackend()
    backend.set("k", frame, {"execution_time": 1.0})
    hit = backend.get("k")[1]
    assert _layout(hit) == _layout(frame)
    assert hit.values.flags.c_contiguous == frame.values.flags.c_contiguous


def test_row_wise_numpy_on_a_hit_is_bit_identical():
    frame = _pivoted()
    backend = InMemoryBackend()
    backend.set("k", frame, {"execution_time": 1.0})
    hit = backend.get("k")[1]
    weights = np.random.default_rng(2).random(frame.shape[1])
    plain = [weights @ frame.values[i] for i in range(len(frame))]
    served = [weights @ hit.values[i] for i in range(len(hit))]
    assert plain == served


def test_a_hit_inside_a_dict_keeps_its_memory_order():
    frame = _pivoted()
    backend = InMemoryBackend()
    backend.set("k", {"close": frame}, {"execution_time": 1.0})
    hit = backend.get("k")[1]["close"]
    assert hit.values.flags.c_contiguous
