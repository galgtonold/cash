"""Which runs of a cell's assignments a restore of a later version may jump.

See ``UpstreamChecker.plan_cell_run`` (round 25, r25s2). A jump skips or
restores statements, so every write they make must land in an object the run
itself made: ``y = x; y[0] += 5`` changes ``x`` too, and ``v = arr[1:]; v += 1``
changes ``arr``.
"""

import ast

import pytest

from cash.notebook.ipython.cell_executor import _jumpable_runs


def _runs(src):
    return _jumpable_runs(ast.parse(src).body, src, lambda code: False)


@pytest.mark.parametrize(
    "src",
    [
        "sales = raw[raw['a'] >= 0]\nsales = sales.drop_duplicates()\nsales['t'] = sales['a'] * 2",
        "df = load()\ndf['x'] = 1\nflag = df['x'] > 0\ndf['y'] = flag.astype(int)",
        "total = a + b\ntotal = total * 2",
    ],
)
def test_a_run_rebuilding_its_own_objects_can_jump(src):
    assert _runs(src) == {0: len(ast.parse(src).body)}


@pytest.mark.parametrize(
    "src",
    [
        "y = x\ny[0] += 5\ny = y + [1]",  # alias of an outside list
        "v = arr[1:]\nv += 1\nv = v * 2",  # a numpy view
        "v = arr[::2]\nv[0] = 99\nv = v + 1",
        "df2 = df\ndf2['a'] += 100\ndf2 = df2.copy()",
        "col = df['a']\ncol[0] = 1\ncol = col + 1",  # a column may be a view
        "m = np.asarray(arr)\nm[0] = 1\nm = m + 1",  # may be arr itself
        "sales = sales.dropna()\nsales = sales.copy()",  # reads what it rebuilds first
        "a = 1\nb = 2",  # nothing rebuilt
    ],
)
def test_a_run_that_may_write_outside_itself_does_not_jump(src):
    assert _runs(src) == {}
