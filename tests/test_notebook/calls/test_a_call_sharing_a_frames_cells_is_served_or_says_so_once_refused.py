"""A call whose result shares a frame's cells is told about only when the
statement around it is refused too, and served frames keep the frame's
own objects, a categorical's among them.

``w = mk(df)`` with ``mk`` handing back its argument: the call's own entry
is refused, but the statement stores ``w`` and serves it. The warning
(``NOTEBOOK-CELLS-SHARED``) came from the call before the statement had
decided, and said "runs every time" of a cell that was served. It is said
once the statement is refused as well, and also for a cell the result
holds outside a frame (``{'f': v.iloc[4]}``) or an item of
``df['o'].values``.

``f = flt(df)`` returning ``d[d.x > 0]`` of a ten-row frame ran every time:
the statement looked for the frame sharing its cells by address only past
64 such cells, and the walk stops at a frame's slots. A categorical's
objects were read as cells everywhere, and pandas' own index of them holds
each too: a frame with one was refused (``a = ad(df)``) and a filter of it
warned it could not find every holder. They are read where a result holds
them (``d.groupby('c').sum()``), and a frame sharing them joins the
statement's group, like one sharing cells.
"""

from __future__ import annotations

import warnings

from tests._cell_driver import run_cash_cell


class K:
    """An object with state of its own, a categorical's category below. A
    class of this module, not of the cell: the mock shell's cells are no
    module a pickle can find a class in, and an entry that cannot be pickled
    keeps the frame's own cells, which the next share check counts."""

    def __init__(self, v):
        self.v = [v]


SETUP = (
    "import time\nimport numpy as np, pandas as pd\n"
    f"from {__name__} import K\n"
    "df = pd.DataFrame({'x': np.arange(-3, 7), 'o': [{'i': i} for i in range(10)],"
    " 'c': pd.Categorical([K(i % 3) for i in range(10)])})"
)
FUNCTIONS = (
    "def mk(v):\n    time.sleep(0.3)\n    return v\n"
    "def mkc(v):\n    time.sleep(0.3)\n    return v.copy()\n"
    "def flt(d):\n    time.sleep(0.3)\n    return d[d.x > 0]\n"
    "def ad(d):\n    time.sleep(0.3)\n    return d.assign(y=d.x * 2)\n"
    "def agg(d):\n    time.sleep(0.3)\n    return d.groupby('c', observed=True)['x'].sum().reset_index()\n"
    "def first(v):\n    time.sleep(0.3)\n    return {'f': v.iloc[4]}\n"
)


def _rounds(cash_magics, last, rounds=2):
    """The warnings each round of the whole notebook gave, by code."""
    cells = [SETUP, FUNCTIONS, last]
    told = []
    for _ in range(rounds):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
        told.append(
            [str(w.message) for w in caught if "CELLS-SHARED" in str(w.message) or "UNCHECKED" in str(w.message)]
        )
    return told


def test_a_call_handing_back_its_argument_is_served_without_a_warning(cash_magics, mock_shell):
    import pandas as pd

    for last in ("w = mk(df)", "w = mk(df['o'])"):
        told = _rounds(cash_magics, last)
        # pandas 2 hands out its own cached column for ``df['o']``: a live
        # alias of ``df``, which the statement does not store.
        if int(pd.__version__.split(".")[0]) >= 3 or last == "w = mk(df)":
            assert told == [[], []], last
        ns = mock_shell.user_ns
        assert (
            ns["w"]["o"].iloc[4] is ns["df"]["o"].iloc[4]
            if last == "w = mk(df)"
            else ns["w"].iloc[4] is ns["df"]["o"].iloc[4]
        )


def test_a_small_filter_and_a_frame_with_a_categorical_are_kept_whole(cash_magics, mock_shell):
    for last, picked in (("f = flt(df)", "f"), ("a = ad(df)", "a")):
        assert _rounds(cash_magics, last) == [[], []], last
        ns = mock_shell.user_ns
        row = ns[picked].index[1]
        assert ns[picked]["o"].loc[row] is ns["df"]["o"].loc[row]
        assert ns[picked]["c"].loc[row] is ns["df"]["c"].loc[row]


def test_a_filter_of_a_frame_with_a_categorical_finds_its_holders(cash_magics, mock_shell):
    assert _rounds(cash_magics, "df2 = df[df.x > 0]") == [[], []]
    ns = mock_shell.user_ns
    row = ns["df2"].index[1]
    assert ns["df2"]["c"].loc[row] is ns["df"]["c"].loc[row]
    assert ns["df2"]["o"].loc[row] is ns["df"]["o"].loc[row]


def test_a_result_holding_a_categoricals_objects_holds_the_frames_own(cash_magics, mock_shell):
    for last in ("a = agg(df)", "a = mkc(df['c'])"):
        _rounds(cash_magics, last)
        ns = mock_shell.user_ns
        got = ns["a"]["c"].iloc[1] if last == "a = agg(df)" else ns["a"].iloc[1]
        assert got is ns["df"]["c"].iloc[1], last


def test_cells_held_outside_a_frame_are_never_copies_and_the_refusal_is_said(cash_magics, mock_shell):
    for last, got in (("w = first(df['o'])", "w['f']"), ("w = mkc(df['o'].values)", "w[4]")):
        told = _rounds(cash_magics, last)
        assert any("NOTEBOOK-CELLS-SHARED" in message for message in told[0]), (last, told)
        ns = mock_shell.user_ns
        assert eval(got, ns) is ns["df"]["o"].iloc[4], last
