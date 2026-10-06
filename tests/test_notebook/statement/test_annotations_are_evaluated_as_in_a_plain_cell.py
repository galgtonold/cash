"""A cell's annotations are evaluated unless the notebook asked otherwise.

Cash compiles each statement itself, and ``compile`` inherits the
``from __future__ import annotations`` of the module calling it. So
``x: Undefined = 1`` and ``def f(a: Undefined)`` ran without the NameError a
plain cell raises, which made a replayed notebook diverge from its plain run.
The notebook's own ``from __future__`` imports still apply to what follows.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell


def test_an_annotated_assignment_with_an_unknown_name_raises(cash_magics, mock_shell):
    with pytest.raises(NameError, match="Undefined"):
        run_cash_cell(cash_magics, "y: Undefined = 3")
    assert mock_shell.user_ns["y"] == 3, "the value is bound before the annotation is evaluated, as in plain Python"


def test_a_function_annotation_with_an_unknown_name_raises(cash_magics, mock_shell):
    with pytest.raises(NameError, match="Nope"):
        run_cash_cell(cash_magics, "def f(a: Nope):\n    return 1")
    assert "f" not in mock_shell.user_ns


@pytest.fixture
def ipython_shell():
    """The shell whose compiler remembers a cell's ``from __future__`` imports."""
    from IPython.core.interactiveshell import InteractiveShell

    shell = InteractiveShell.instance()
    before = shell.compile.flags
    yield shell
    shell.compile.flags = before


def test_the_notebooks_own_future_import_leaves_annotations_unevaluated(cash_magics, mock_shell, ipython_shell):
    run_cash_cell(cash_magics, "from __future__ import annotations")
    run_cash_cell(cash_magics, "def g(a: Nope):\n    return 2")
    assert mock_shell.user_ns["g"](1) == 2
    assert mock_shell.user_ns["g"].__annotations__ == {"a": "Nope"}
