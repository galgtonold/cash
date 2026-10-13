"""Writes into what numpy, pandas and polars hand back, and polars' ``sort``.

A numpy / pandas / polars function builds a new object, so changing it is
local to the function: ``load = np.bincount(...); load[c] -= d`` and
``counts = pd.crosstab(...); counts.columns = [...]`` were reported as side
effects. polars' ``df.sort("ts")`` returns a new frame, and was reported as a
write method. Each silenced pattern has a control that must still warn: a
write into an argument, a global, or a view or pass-through of either.
"""

from __future__ import annotations

import importlib.util
import textwrap

import pytest

from cash.analysis.purity_analyzer import PurityAnalyzer
from tests._scripts import run_python

np = pytest.importorskip("numpy")

pytestmark = pytest.mark.core

LOADS = np.zeros(4)


def _mutations(fn):
    issues = PurityAnalyzer().analyze(fn).issues
    return [i for i in issues if i.kind in ("scope_mutation", "impure_call")]


# -- numpy ---------------------------------------------------------------------


def rebalance(labels, demand, n):
    labels = labels.copy()
    load = np.bincount(labels, weights=demand, minlength=n)
    for s in range(len(labels)):
        c = labels[s]
        if load[c] > 5:
            labels[s] = (c + 1) % n
            load[c] -= demand[s]
    return labels, load


def chained_result(x):
    h = np.histogram(x, bins=3)[0].astype(float)
    h[0] = 0
    return h


def writes_through_asarray(x):
    a = np.asarray(x)
    a[0] = 1
    return a


def writes_through_out(x, y):
    r = np.add(x, y, out=x)
    r[0] = 1
    return r


def writes_a_global(i):
    LOADS[i] += 1
    return i


def writes_a_reshaped_argument(x):
    v = np.reshape(x, -1)
    v[0] = 1
    return v


def test_writing_into_a_new_numpy_array_is_local():
    assert not _mutations(rebalance), _mutations(rebalance)
    assert not _mutations(chained_result), _mutations(chained_result)


@pytest.mark.parametrize(
    "fn",
    [writes_through_asarray, writes_through_out, writes_a_global, writes_a_reshaped_argument],
    ids=["asarray", "out=", "global", "reshape-view"],
)
def test_writing_into_an_argument_or_global_through_numpy_still_warns(fn):
    assert _mutations(fn)


# -- pandas --------------------------------------------------------------------


def test_renaming_the_columns_of_a_new_frame_is_local():
    pd = pytest.importorskip("pandas")

    def event_counts(ev):
        counts = pd.crosstab(ev["c"], ev["k"])
        counts.columns = [f"n_{c}" for c in counts.columns]
        return counts

    def renames_the_argument(ev):
        ev.columns = ["a", "b"]
        return ev

    assert not _mutations(event_counts), _mutations(event_counts)
    assert _mutations(renames_the_argument)


# -- sort: polars' returns a new frame, list's sorts in place ------------------


def sorts_a_list_argument(rows):
    rows.sort()
    return rows


def sorts_a_global_list():
    GLOBAL_ROWS.sort()
    return len(GLOBAL_ROWS)


GLOBAL_ROWS: list = [3, 1]


def sorts_in_place_on_request(df):
    out = df.sort("x", in_place=True)
    return out


def returns_a_sorted_copy(df):
    out = df.filter(df["x"] > 0)
    return out.sort("x")


def sorts_at_the_end_of_a_chain(df):
    return df.group_by("h").agg(n=1).sort("h")


def test_a_sort_whose_result_is_used_is_not_a_write():
    assert not _mutations(returns_a_sorted_copy), _mutations(returns_a_sorted_copy)
    assert not _mutations(sorts_at_the_end_of_a_chain), _mutations(sorts_at_the_end_of_a_chain)


@pytest.mark.parametrize(
    "fn",
    [sorts_a_list_argument, sorts_a_global_list, sorts_in_place_on_request],
    ids=["list-argument", "global-list", "in_place=True"],
)
def test_a_sort_in_place_still_warns(fn):
    assert _mutations(fn)


# -- a waiver on the line of a chained call -------------------------------------


def test_a_waiver_on_the_line_of_a_chained_call_is_honoured(tmp_path):
    """The finding names the line the method is on, and a waiver there (or
    on the line the statement opens on) covers it; a chain without one
    still warns, and names that line."""
    module = tmp_path / "chain_mod.py"
    module.write_text(
        textwrap.dedent(
            """\
            def unwaived(df, extra):
                (df.attrs
                    .update(extra))
                return df

            def waived_on_the_method(df, extra):
                (df.attrs
                    .update(extra))  # @cash:assume-safe
                return df

            def waived_on_the_opening_line(df, extra):
                (df.attrs  # @cash:assume-safe
                    .update(extra))
                return df
            """
        ),
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("chain_mod", module)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    found = _mutations(mod.unwaived)
    assert [i.line for i in found] == [3], found
    assert not _mutations(mod.waived_on_the_method)
    assert not _mutations(mod.waived_on_the_opening_line)


def test_a_polars_pipeline_runs_without_a_warning(tmp_path):
    """End to end, as a script user meets it: polars' ``sort`` in a cached
    function, at the end of a chain and returned."""
    pytest.importorskip("polars")
    script = tmp_path / "etl.py"
    script.write_text(
        textwrap.dedent(
            """\
            import warnings
            warnings.simplefilter("always")
            import polars as pl
            import cash

            @cash.cache
            def clean(raw):
                out = raw.filter(pl.col("x") > 0)
                return out.sort("x")

            @cash.cache
            def hourly(df):
                return (
                    df.group_by("h")
                    .agg(pl.len().alias("n"))
                    .sort("h")
                )

            raw = pl.DataFrame({"x": [1, -1, 3], "h": [1, 2, 1]})
            print(clean(raw).shape, hourly(raw).shape)
            """
        ),
        encoding="utf-8",
    )
    result = run_python(script, cwd=tmp_path)
    assert "(2, 2) (2, 2)" in result.stdout
    assert "IMPURE-SIDE-EFFECTS" not in result.stderr, result.stderr
    assert "UNWALKABLE" not in result.stderr, result.stderr
