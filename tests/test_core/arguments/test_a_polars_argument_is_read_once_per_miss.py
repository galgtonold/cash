"""A big polars argument is read once per miss, not again after the body.

The in-place-change check hashed every argument again after the body:
``clean(raw)`` on a 4.2M-row log frame read it twice per miss, 0.7 s each.
The key's digest is kept for the check while the frame's buffers stay where
they were (`CallPolarsDigests`); a write through polars moves them, and is
still seen.
"""

from __future__ import annotations

import functools
import warnings

import numpy as np
import pytest

pl = pytest.importorskip("polars")

from cash import Cash
from cash.decorator import arg_hashing

ROWS = 300_000


def _frame():
    rng = np.random.default_rng(0)
    return pl.DataFrame({"a": rng.random(ROWS), "s": [f"row{i % 97}" for i in range(ROWS)]})


def _counting_hashes(monkeypatch):
    read: list = []
    real = arg_hashing.builtin_hash

    def counting(value):
        if isinstance(value, pl.DataFrame):
            read.append(1)
        return real(value)

    monkeypatch.setattr(arg_hashing, "builtin_hash", counting)
    return read


def test_a_miss_reads_the_frame_once(tmp_path, monkeypatch):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ran: list = []

    @c.cache
    def total(df):
        ran.append(1)
        return float(df["a"].sum())

    df = _frame()
    read = _counting_hashes(monkeypatch)
    first = total(df)
    assert len(read) == 1
    assert total(df) == first
    assert ran == [1]


def test_a_write_through_polars_in_the_body_is_still_seen(tmp_path, monkeypatch):
    c = Cash(cache_dir=str(tmp_path / ".cash"), register_magic=False)
    ran: list = []

    @c.cache
    def grow(df):
        ran.append(1)
        # Out of the static pass's sight: only the check after the body sees it.
        functools.reduce(pl.DataFrame.extend, [df.head(1)], df)
        return df.height

    df = _frame()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert grow(df) == ROWS + 1
        assert grow(df) == ROWS + 2
    assert ran == [1, 1]
    assert any("argument mutation" in str(w.message) or "in place" in str(w.message) for w in record)
