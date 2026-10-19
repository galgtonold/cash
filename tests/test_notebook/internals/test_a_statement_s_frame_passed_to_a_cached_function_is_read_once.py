"""A frame a statement made, handed to a cached function, is read once.

``s = signal(close, lb)`` then ``run_backtest(close, s, ...)`` in the loop
under it: the statement hashed ``s`` for its lineage, then the call hashed it
again as an argument, every byte of a 3,000-column frame per lookback. A
statement's output carries a lineage tag, and the argument hash skipped the
copy-on-write memo for a tagged frame. It now asks the memo first, whose
check is exact, so the key is the one a full read gives.

Each use of a ``frozen=True`` result as an argument also named the dtype of
every column (``frozen_shape``), ~10 ms for 3,000 columns: each distinct
dtype is named once now, the same names.
"""

from __future__ import annotations

import pytest

from cash.decorator import arg_hashing, frozen
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

if int(pd.__version__.split(".")[0]) < 3:  # pragma: no cover - the memo needs copy-on-write
    pytest.skip("copy-on-write is pandas 3's only mode", allow_module_level=True)


@pytest.fixture
def stores_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0


@pytest.fixture
def argument_reads(monkeypatch):
    """The wide frames read in full through the shared memo or the argument
    hash (`arg_hashing.builtin_hash`)."""
    seen = []
    real = arg_hashing.builtin_hash

    def counting(value, *a, **k):
        if isinstance(value, pd.DataFrame) and value.shape[1] >= 300:
            seen.append(value.shape)
        return real(value, *a, **k)

    monkeypatch.setattr(arg_hashing, "builtin_hash", counting)
    return seen


SETUP = (
    "import numpy as np, pandas as pd, cash\n"
    "close = pd.DataFrame(np.random.default_rng(0).random((1000, 300)))\n"
    "@cash.cache\n"
    "def run(close, s, n):\n"
    "    return float((close * s).iloc[:, :n].sum().sum())\n"
)

LOOP = (
    "out = []\n"
    "for lb in [5, 10]:\n"
    "    s = close.rolling(lb).mean()\n"
    "    for n in [10, 20, 30]:\n"
    "        out.append(run(close, s, n))\n"
)


def test_a_statement_s_frame_is_not_read_again_as_an_argument(
    cash_magics, mock_shell, stores_everything, argument_reads
):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    argument_reads.clear()
    run_cash_cell(cash_magics, LOOP)
    close, out = mock_shell.user_ns["close"], mock_shell.user_ns["out"]
    want = [float((close * close.rolling(lb).mean()).iloc[:, :n].sum().sum()) for lb in [5, 10] for n in [10, 20, 30]]
    assert out == pytest.approx(want)
    # Each `s` read once, by its statement's lineage; `close` and both `s` as
    # arguments from the memo (it was five: the arguments read each again).
    assert len(argument_reads) == 2, argument_reads


def test_the_memo_s_digest_of_a_tagged_frame_is_a_full_read_s(cash_magics, mock_shell, stores_everything):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, "s = close.rolling(5).mean()")
    s = mock_shell.user_ns["s"]
    memo = arg_hashing.FRAME_DIGESTS.lookup(s)
    assert memo is not None, "the statement's read of its output was not kept"
    assert memo == arg_hashing.builtin_hash(s)


def test_a_wide_frame_s_shape_names_each_dtype_as_str_does():
    wide = pd.concat(
        [
            pd.DataFrame(np.zeros((3, 400))),
            pd.DataFrame(np.zeros((3, 50), dtype="int32")),
            pd.DataFrame({"s": ["a"] * 3, "c": pd.Categorical(["x", "y", "x"])}),
            pd.DataFrame({"t": pd.date_range("2024", periods=3, tz="UTC")}),
        ],
        axis=1,
    )
    assert frozen.frozen_shape(wide) == ("shaped", wide.shape, tuple(str(d) for d in wide.dtypes))
