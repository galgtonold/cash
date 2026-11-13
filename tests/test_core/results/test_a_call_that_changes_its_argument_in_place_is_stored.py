"""A cached call that changes an argument in place is stored with the state
it left the argument in, and a hit writes that state back into the argument.

``W, H = fit_nmf(X, k)`` with ``fit_nmf`` writing into ``X`` was never
stored: a hit would have returned ``W`` and ``H`` and left the caller's ``X``
as it was. It is stored with the state it left the arguments a hit could
write into in after the body. A hit checks that the argument handed in is in
the state the entry was stored from, then writes the stored state into that
very object: the caller's other names for it, and views of an array, see the
change as after a plain call.

Only NumPy arrays, pandas frames and Series whose axes and dtypes the call
kept, and lists, dicts and sets of plain values are written back. Anything
else still runs every time.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

needs_trace = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="what a call ran is recorded with sys.monitoring, Python 3.12+; below it a call that changes its "
    "argument in place runs every time, with a warning",
)


def _change(v):
    """What the cached functions below do to their argument."""
    if isinstance(v, np.ndarray):
        np.clip(v, 0.25, None, out=v)
        return float(v.sum())
    if isinstance(v, pd.DataFrame):
        v["a"] *= 2
        v.loc[0, "b"] = "z"
        return float(v["a"].sum())
    if isinstance(v, pd.Series):
        v.iloc[1] = 99.0
        return float(v.sum())
    if isinstance(v, list):
        v.sort(reverse=True)
        v.append(-1)
        return len(v)
    if isinstance(v, dict):
        v["n"] = v.get("n", 0) + 7
        del v["gone"]
        return len(v)
    if isinstance(v, set):
        v.add("new")
        v.discard("a")
        return len(v)
    v.n = getattr(v, "n", 0) + 7
    return 1


KINDS = {
    "array": lambda: np.arange(12, dtype=float).reshape(3, 4) / 20,
    "frame": lambda: pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "w"]}),
    "series": lambda: pd.Series([1.0, 2.0, 3.0], index=["p", "q", "r"], name="s"),
    "list": lambda: [3, 1, 2],
    "dict": lambda: {"n": 1, "gone": 2, "k": "v"},
    "set": lambda: {"a", "b", "c"},
}


def _same(a, b) -> bool:
    if isinstance(a, (pd.DataFrame, pd.Series)):
        return type(a) is type(b) and a.equals(b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and a.tolist() == b.tolist()
    return type(a) is type(b) and a == b


@needs_trace
@pytest.mark.parametrize("kind", list(KINDS))
def test_the_state_left_in_the_argument_is_written_back_on_a_hit(disk_cash, kind):
    runs = []

    @disk_cash.cache
    def work(v):
        runs.append(1)
        return _change(v)

    make = KINDS[kind]
    work(make())  # stored, with the state it left the argument in
    warm = len(runs)
    if kind == "frame":
        # The change is made in a helper, which reading `work` does not
        # show, so the first call ran with cash's frame-hash memo holding
        # the frame's blocks; pandas then copies a column before writing
        # it, and whether it would have without cash cannot be told. The
        # next call lets go of the frame first and is stored.
        work(make())
    assert warm == 1
    warm = len(runs)
    v = make()
    alias = v
    result = work(v)
    assert len(runs) == warm, f"the {kind} the call changes in place sent it back to run"
    expected = make()
    assert result == _change(expected)
    assert alias is v
    assert _same(v, expected), (v, expected)


@needs_trace
def test_a_view_of_the_argument_sees_the_write(disk_cash):
    """The state goes into the array's memory, which its views share."""
    runs = []

    @disk_cash.cache
    def fit_nmf(X, k):
        runs.append(1)
        np.clip(X, 0.25, None, out=X)
        return X[:k].copy(), float(X.sum()) * k

    fit_nmf(np.arange(12, dtype=float).reshape(3, 4) / 20, 2)
    base = np.arange(12, dtype=float) / 20
    X = base.reshape(3, 4)
    row = X[0]
    W, H = fit_nmf(X, 2)
    assert len(runs) == 1
    assert np.shares_memory(X, base)
    assert base.min() == 0.25 and row.tolist() == X[0].tolist() == [0.25, 0.25, 0.25, 0.25]
    assert W.tolist() == X[:2].tolist()


def test_an_argument_in_another_state_misses(disk_cash):
    runs = []

    @disk_cash.cache
    def work(v):
        runs.append(1)
        return _change(v)

    work(np.zeros(4))
    v = np.ones(4)
    assert work(v) == 4.0
    assert len(runs) == 2


def test_a_key_function_keeps_it_running_every_time(disk_cash):
    """``key=`` says how calls differ, not what state the argument is in:
    no state is written back under it."""
    runs = []

    @disk_cash.cache(key=lambda v: len(v))
    def work(v):
        runs.append(1)
        return _change(v)

    for _ in range(3):
        v = np.zeros(4)
        work(v)
        assert v.tolist() == [0.25] * 4
    assert len(runs) == 3


class Box:
    pass


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(Box, id="a class instance"),
        pytest.param(lambda: [[3], [1], [2]], id="a list of lists"),
        pytest.param(lambda: np.array([1, "a", None], dtype=object), id="an array of objects"),
    ],
)
def test_what_a_write_cannot_restore_runs_every_time(disk_cash, make):
    runs = []

    @disk_cash.cache
    def work(v):
        runs.append(1)
        if isinstance(v, np.ndarray):
            v[2] = "filled"
            return 1
        return _change(v)

    for _ in range(3):
        work(make())
    assert len(runs) == 3


def test_a_result_sharing_memory_with_the_argument_is_not_stored(disk_cash):
    """``return X[:2]``: a hit would hand back a copy that no longer
    shares the argument's memory."""
    runs = []

    @disk_cash.cache
    def work(X):
        runs.append(1)
        X *= 2
        return X[:2]

    for _ in range(3):
        X = np.arange(4.0)
        head = work(X)
        X[0] = 9.0
        assert head[0] == 9.0
    assert len(runs) == 3


def test_a_frame_given_a_new_column_runs_every_time(disk_cash):
    runs = []

    @disk_cash.cache
    def add(df):
        runs.append(1)
        df["c"] = df["a"] * 3
        return 1

    for _ in range(3):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        add(df)
        assert list(df.columns) == ["a", "c"]
    assert len(runs) == 3
