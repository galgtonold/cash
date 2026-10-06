"""IPython syntax in a cell above is read as IPython runs it, not as Python.

A cell magic other than ``%%time``/``%%capture`` runs no code in the user's
namespace: ``%%script false`` skips its body, ``%%writefile`` writes it to a
file. The upstream check read the body as Python, recorded it as the producer
of the names it assigns and re-ran it in the kernel, so a Run All bound
``threshold = 0.9`` from a disabled cell. ``files = !ls`` and a ``%%bash``
cell were reported as syntax errors (NOTEBOOK-CELL-SYNTAX).
"""

from __future__ import annotations

import warnings

import pytest

from cash import CashUpstreamSyntaxWarning
from cash.analysis.code_analyzer import CodeAnalyzer
from tests._cell_driver import run_cash_cell


@pytest.mark.parametrize(
    "cell",
    [
        "%%script false --no-raise-error\nthreshold = 0.9",
        "%%writefile cfg.py\nRATE = 99",
        "%%timeit\nn = sum(range(10))",
        "%%bash\necho hi",
        "\n%%html\n<b>hi</b>",
    ],
)
def test_a_cell_magic_that_runs_no_python_strips_to_nothing(cell):
    assert CodeAnalyzer.strip_magics(cell) == ""


@pytest.mark.parametrize("magic", ["%%time", "%%capture out"])
def test_a_cell_magic_running_its_body_in_the_namespace_keeps_it(magic):
    assert CodeAnalyzer.strip_magics(f"{magic}\nx = 1\ny = x + 1") == "x = 1\ny = x + 1"


@pytest.mark.parametrize("line", ["files = !echo hi", "t = %time f()", "a, b = !ls"])
def test_a_magic_assignment_is_a_magic_line(line):
    assert CodeAnalyzer.strip_magics(f"{line}\nz = 1") == "z = 1"


def test_a_comparison_is_not_mistaken_for_a_magic_assignment():
    code = "if x == 1:\n    y = 2\n%time f()"
    assert CodeAnalyzer.strip_magics(code) == "if x == 1:\n    y = 2"


def test_a_disabled_cell_does_not_rebind_what_it_assigns(cash_magics, mock_shell):
    cells = [
        "threshold = 0.5\nRATE = 1",
        "%%script false --no-raise-error\nthreshold = 0.9",
        "%%writefile cfg.py\nRATE = 99",
        "y = (threshold, RATE * 2)",
    ]
    run_cash_cell(cash_magics, cells[0], cells=cells)
    # Cells 2 and 3 run in IPython and bind nothing.
    run_cash_cell(cash_magics, cells[3], cells=cells)

    assert mock_shell.user_ns["y"] == (0.5, 2)
    assert mock_shell.user_ns["threshold"] == 0.5


def test_valid_ipython_cells_above_are_not_syntax_errors(cash_magics, mock_shell):
    cells = ["files = !echo hi", "%%bash\necho hi", "t = %time sum([1])", "n = 1"]
    mock_shell.user_ns.update(files=["hi"], t=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[3], cells=cells)

    assert not [w for w in caught if issubclass(w.category, CashUpstreamSyntaxWarning)]
    assert mock_shell.user_ns["n"] == 1
