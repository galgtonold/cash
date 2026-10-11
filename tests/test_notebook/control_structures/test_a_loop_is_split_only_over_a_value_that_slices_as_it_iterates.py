"""A loop is split only over a value whose slices iterate what it iterates.

A split runs ``for x in v[:k]`` then ``for x in v[k:]``. That is the loop
only when slicing ``v`` cuts what iterating it walks. A ``DataFrame``
iterates its columns but slices its rows, so each half of ``for col in df:``
over a 60-row frame iterated all 8 columns, and the loop ran every column
twice from the second Run All on, also after a restart.
"""

import ast
from unittest.mock import MagicMock

import pytest

from cash.notebook.control_structures.split_policy import LoopSplitPolicy

LOOP = ast.parse("for x in v:\n    out.append(x)").body[0]


def _eligible(value):
    return LoopSplitPolicy(MagicMock()).eligible(LOOP, value, {"v": value})


@pytest.mark.parametrize(
    "value",
    [list(range(60)), tuple(range(60)), range(60), "x" * 60, b"x" * 60],
    ids=["list", "tuple", "range", "str", "bytes"],
)
def test_a_builtin_sequence_is_eligible(value):
    assert _eligible(value) == 60


def test_an_array_is_eligible_along_its_first_axis():
    from cash.notebook.control_structures.split_policy import slices_as_it_iterates

    np = pytest.importorskip("numpy")
    assert _eligible(np.zeros((60, 3))) == 60
    assert not slices_as_it_iterates(np.float64(1.0))


def test_a_frame_is_not_split():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({c: range(60) for c in "abcdefgh"})
    assert _eligible(df) is None, "a frame iterates its columns but slices its rows"


def test_a_series_and_an_index_are_not_split():
    pd = pytest.importorskip("pandas")
    series = pd.Series(range(60), index=range(100, 160))
    assert _eligible(series) is None
    assert _eligible(series.index) is None


def test_a_subclass_of_list_is_not_split():
    class Rows(list):
        def __iter__(self):
            return iter(reversed(list.__iter__(self)))

    assert _eligible(Rows(range(60))) is None


def test_a_verdict_recorded_before_the_rule_is_dropped(tmp_path):
    """The simulation applies a stored verdict without seeing the value, so a
    verdict recorded for ``for col in df:`` would model halves the runtime no
    longer runs."""
    import json

    from cash.notebook.loop_split import LoopSplitStore

    (tmp_path / "_loop_split.json").write_text(json.dumps({"version": 1, "splits": {"abc": 5}}), encoding="utf-8")
    assert LoopSplitStore(str(tmp_path)).get("abc") is None
