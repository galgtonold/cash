"""A function handed to a call writes its globals again when the statement re-runs.

``s.apply(f)`` and ``list(map(f, rows))`` name ``f`` without calling it, so
only calls by name were checked for the globals a callee writes. After
``pairs = []`` ran again the statement was served from the cache, ``f`` never
ran, and ``pairs`` stayed empty where a plain kernel has it full (found
replaying a student's notebook of the JuNE dataset, 'expert_5' step 205).
"""

from __future__ import annotations

import ast
import warnings

import pytest

from cash.analysis.callee_effects import callee_global_mutations
from tests._cell_driver import run_cash_cell

SOURCES = {"f": "def f(row):\n    SEEN.append(row)\n", "g": "def g(row):\n    return row\n"}


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        ("x = list(map(f, rows))", {"SEEN"}),
        ("x = s.apply(f)", {"SEEN"}),
        ("x = s.apply(func=f)", {"SEEN"}),
        ("x = list(map(g, rows))", set()),
        ("x = list(map(h, rows))", set()),
    ],
)
def test_a_handed_function_counts_like_a_called_one(statement, expected):
    assert callee_global_mutations(ast.parse(statement), SOURCES.get) == expected


def test_a_helper_handing_a_function_on_counts_too():
    sources = {"f": SOURCES["f"], "helper": "def helper(rows):\n    return list(map(f, rows))\n"}
    assert callee_global_mutations(ast.parse("x = helper(rows)"), sources.get) == {"SEEN"}


@pytest.mark.parametrize("spelling", ["list(map(f, rows))", "pd.Series(rows).map(f)"])
def test_a_rerun_of_the_statement_writes_again(cash_magics, capsys, spelling):
    cells = [
        "import pandas as pd\nrows = [list(range(5)) for _ in range(3000)]",
        "def f(acts):\n    for i in range(len(acts) - 1):\n        pairs.append((acts[i], acts[i + 1]))",
        f"pairs = []\nx = {spelling}\nprint(len(pairs))",
    ]
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cell in cells:
            run_cash_cell(cash_magics, cell)
        capsys.readouterr()
        run_cash_cell(cash_magics, cells[2])
    counts = [word for word in capsys.readouterr().out.split() if word.isdigit()]
    assert counts[-1] == "12000"
