"""Building a value with ordinary pandas/numpy/builtins is not a side effect.

Found while attacking the decorator before round 26, on the first realistic
pipeline written: a groupby aggregation assigned to a local and then given a
column warned ``IMPURE-SIDE-EFFECTS``. The escape analysis knows a local bound
to a fresh allocation cannot reach caller state, but its list of "returns a new
object" spellings missed the aggregations, the numpy builders and ``sorted()``.
"""

from __future__ import annotations

import warnings

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from cash import Cash
from cash.backends import InMemoryBackend

FRAME = pd.DataFrame({"store": [1, 1, 2], "units": [1.0, 2.0, 3.0]})
ARR = np.arange(6.0)


@pytest.fixture
def cash():
    return Cash(backend=InMemoryBackend(), register_magic=False)


def _impure(fn, arg):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        fn(arg)
    return [str(w.message) for w in seen if "IMPURE" in str(w.message)]


def test_a_groupby_aggregation_is_a_new_frame(cash):
    @cash.cache
    def totals(df):
        g = df.groupby("store", as_index=False)["units"].sum()
        g["roll"] = g["units"] * 2
        return g

    assert not _impure(totals, FRAME)


def test_a_transform_and_an_apply_are_new(cash):
    @cash.cache
    def shaped(df):
        g = df.groupby("store", as_index=False)["units"].mean()
        g["scaled"] = g["units"] / g["units"].max()
        return g

    assert not _impure(shaped, FRAME)


def test_numpy_builders_are_new(cash):
    @cash.cache
    def joined(a):
        out = np.concatenate([a, a])
        out[0] = -1.0
        return float(out.sum())

    assert not _impure(joined, ARR)


def test_sorted_returns_a_new_list(cash):
    @cash.cache
    def ordered(df):
        rows = sorted(df.to_dict("records"), key=lambda r: r["units"])
        rows[0]["units"] = 0.0
        return rows

    assert not _impure(ordered, FRAME)


def test_a_real_argument_mutation_still_warns(cash):
    @cash.cache
    def stamp(df):
        df["seen"] = 1
        return len(df)

    assert _impure(stamp, FRAME.copy()), "a write into the caller's frame must warn"


def test_a_real_global_mutation_still_warns(cash):
    counter: dict = {}

    @cash.cache
    def bump(n):
        counter[n] = counter.get(n, 0) + 1
        return counter[n]

    assert _impure(bump, 1), "a write into a captured dict must warn"


def test_logging_through_get_logger_is_a_diagnostic_line(cash):
    """``logger.info(...)`` was recognised; the same call written in one
    expression -- ``logging.getLogger(__name__).info(...)`` -- was not, because
    the receiver is a call rather than a name."""
    import logging

    @cash.cache
    def with_logging(n):
        logging.getLogger(__name__).info("computing %s", n)
        return n * 2

    assert not _impure(with_logging, 3)


def test_a_print_to_stdout_still_says_so(cash):
    """Deliberate, and older than this sweep: stdout may be the program's
    output, and a hit does not reprint it."""

    @cash.cache
    def chatty(n):
        print("working on", n)
        return n * 2

    assert _impure(chatty, 3)
