"""A loop over ``enumerate(np.array(...))`` run as one unit gives what Python gives.

The header is sized from the list the array is built from; the loop then
runs as one unit, which evaluates the header once. These pin that the values,
the loop variables left behind and the bar match plain Python, that an edit
to the list is followed, and that a call building a differently shaped array
is not sized by the list.
"""

from __future__ import annotations

import re

import pytest

from tests._cell_driver import run_cash_cell

np = pytest.importorskip("numpy")

LOOP = (
    "d = []\nfor n, line in enumerate(np.array(lines)):\n"
    "    for i, action in enumerate(np.array(line.split(','))):\n        d.append((n, i, str(action)))"
)


def _plain(setup, loop):
    ns: dict = {}
    exec(setup, ns)
    exec(loop, ns)
    return ns


def test_the_values_match_plain_python_and_follow_an_edit(cash_magics):
    ns = cash_magics.shell.user_ns
    for k in (2, 3):
        setup = f"import numpy as np\nlines = [','.join(['a%d' % j] * {k}) for j in range(2000)]"
        run_cash_cell(cash_magics, setup)
        run_cash_cell(cash_magics, LOOP)
        plain = _plain(setup, LOOP)
        assert ns["d"] == plain["d"]
        assert (ns["n"], ns["i"], str(ns["action"])) == (plain["n"], plain["i"], str(plain["action"]))


def test_an_error_partway_leaves_what_python_leaves(cash_magics):
    ns = cash_magics.shell.user_ns
    setup = "import numpy as np\nlines = list(range(2000))"
    loop = (
        "d = []\nfor n, x in enumerate(np.array(lines)):\n    if x == 1500:\n"
        "        raise ValueError('stop at 1500')\n    d.append(int(x))"
    )
    run_cash_cell(cash_magics, setup)
    with pytest.raises(ValueError, match="stop at 1500"):
        run_cash_cell(cash_magics, loop)
    assert (ns["n"], len(ns["d"])) == (1500, 1500)


def test_one_bar_is_drawn(cash_magics, capsys):
    pytest.importorskip("tqdm")
    ns = cash_magics.shell.user_ns
    run_cash_cell(cash_magics, "from tqdm import tqdm\nimport numpy as np\nlines = ['x'] * 2000")
    capsys.readouterr()
    run_cash_cell(cash_magics, "t = 0\nfor n, line in tqdm(enumerate(np.array(lines))):\n    t += len(line)")
    assert (ns["t"], ns["n"]) == (2000, 1999)
    assert len(re.findall(r"(?<![0-9])0it", capsys.readouterr().err)) == 1  # each bar starts at 0it


@pytest.mark.parametrize(
    "header, expected",
    [("enumerate(np.array(lines, ndmin=2))", 1), ("enumerate(np.array(word))", None)],
    ids=["a leading axis", "a 0-d array"],
)
def test_an_array_that_is_not_as_long_as_its_list_is_not_sized_by_it(header, expected):
    import ast

    from cash.notebook.control_structures.single_unit_policy import estimated_iterations

    ns = {"np": np, "lines": list(range(2000)), "word": "abc"}
    node = ast.parse(f"for n, x in {header}:\n    pass").body[0]
    assert estimated_iterations(node.iter, object(), ns) is None
    if expected is not None:
        assert len(list(eval(header, dict(ns)))) == expected
