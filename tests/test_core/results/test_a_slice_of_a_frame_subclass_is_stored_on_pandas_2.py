"""A slice of a pandas frame subclass is stored and hit on pandas 2, as a
plain frame's is.

pandas 2 (without copy-on-write) puts ``_is_copy``, a weak reference to the
frame a slice was taken from, on ``df[df.a > 2]``, ``df.loc[...]``,
``df.head()`` and the like. Only ``NDFrame._check_setitem_copy`` reads it, to
warn (``SettingWithCopyWarning``) on a later write to the slice; no value of
either frame depends on it, and pickling drops it. A plain ``DataFrame`` is
read for ids by its columns alone, but a subclass of the user's
(``GeoDataFrame``-like) is read attribute by attribute, and the weak
reference to the argument outside the result refused every store:

    STORE-IDENTITY-HELD ... the result holds a weak reference to an object
    outside it, or an object a weak reference elsewhere finds.

and in a notebook ``df2 = df[df.a > 2]`` re-ran every time
(``NOTEBOOK-IDENTITY-HELD``), as did a later statement binding ``df``, which
``df2`` was taken to find by a weak reference.

With ``mode.chained_assignment = "raise"`` the write raises
``SettingWithCopyError`` in plain Python and would not after a hit, so the
result is still refused then. pandas 3 has copy-on-write only and no
``_is_copy`` at all, so these tests skip there: there is nothing to check.
"""

from __future__ import annotations

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from cash.notebook.statement.identity_holders import identity_held_reason

pytestmark = pytest.mark.skipif(
    int(pd.__version__.split(".", 1)[0]) >= 3 or pd.get_option("mode.copy_on_write") is not False,
    reason="pandas 3 (or copy-on-write) keeps no _is_copy weak reference on a slice",
)


class Frame(pd.DataFrame):
    """A frame subclass of the user's, read attribute by attribute."""

    @property
    def _constructor(self):
        return Frame


def _frame() -> Frame:
    return Frame({"a": np.arange(6.0), "b": list("abcdef")})


def _hits(fn) -> int:
    return fn.cache_info()["hits"]


@pytest.mark.parametrize(
    "take",
    [
        pytest.param(lambda f: f[f.a > 2], id="filter"),
        pytest.param(lambda f: f.loc[f.a > 2], id="loc"),
        pytest.param(lambda f: f.head(3), id="head"),
        pytest.param(lambda f: f[["a"]], id="columns"),
    ],
)
def test_a_slice_of_a_frame_subclass_is_stored_and_hit(disk_cash, take):
    @disk_cash.cache
    def part(frame):
        return take(frame)

    frame = _frame()
    expected = take(frame)
    assert expected._is_copy is not None  # the bookkeeping this test is about
    pd.testing.assert_frame_equal(part(frame), expected)
    pd.testing.assert_frame_equal(part(frame), expected)
    assert _hits(part) == 1, part.cache_info()["miss_reasons"]


def test_a_slice_is_refused_while_a_write_to_it_would_raise(disk_cash):
    @disk_cash.cache
    def part(frame):
        return frame[frame.a > 2]

    frame = _frame()
    with pd.option_context("mode.chained_assignment", "raise"):
        plain = frame[frame.a > 2]
        with pytest.raises(pd.errors.SettingWithCopyError):
            plain["c"] = 1  # what a hit's copy, with no _is_copy, would not raise
        with pytest.warns(Warning, match="STORE-IDENTITY-HELD"):
            part(frame)
        part(frame)
    assert _hits(part) == 0


def test_a_notebook_slice_of_a_frame_subclass_holds_no_identity():
    df = _frame()
    df2 = df[df.a > 2]
    assert df2._is_copy is not None
    ns = {"df": df, "df2": df2}
    assert identity_held_reason("df2", df2, ns) is None
    assert identity_held_reason("df", df, ns) is None
    with pd.option_context("mode.chained_assignment", "raise"):
        assert identity_held_reason("df2", df2, ns) is not None


def test_a_column_of_a_frame_subclass_that_writes_back_is_still_refused():
    """A series' ``_cacher`` is no mere warning: through it a write to the
    series changes its parent frame (``DataFrame._maybe_cache_changed``)."""

    class Column(pd.Series):
        @property
        def _constructor(self):
            return Column

    class Table(Frame):
        @property
        def _constructor(self):
            return Table

        @property
        def _constructor_sliced(self):
            return Column

    df = Table({"a": np.arange(6.0), "b": list("abcdef")})
    column = df["a"]
    assert getattr(column, "_cacher", None) is not None
    assert identity_held_reason("s", column, {"df": df, "s": column}) is not None
