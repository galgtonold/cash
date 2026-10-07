"""A cell's annotations are evaluated unless the notebook asked otherwise.

Cash compiles each statement itself, and ``compile`` inherits the
``from __future__ import annotations`` of the module calling it. So
``x: Undefined = 1`` and ``def f(a: Undefined)`` ran without the NameError a
plain cell raises, which made a replayed notebook diverge from its plain run.
The notebook's own ``from __future__`` imports still apply to what follows.

From Python 3.14 annotations are evaluated lazily (PEP 649), so a plain cell
raises nothing for either; cash must then raise nothing too.
"""

from __future__ import annotations

import contextlib

import pytest

from tests._cell_driver import run_cash_cell


def _as_a_plain_cell(source: str) -> contextlib.AbstractContextManager:
    """What a plain cell does with *source*: raise its NameError, or not."""
    try:
        exec(compile(source, "<cell>", "exec", dont_inherit=True), {})
    except NameError as exc:
        return pytest.raises(NameError, match=str(exc.name))
    return contextlib.nullcontext()


def test_an_annotated_assignment_with_an_unknown_name_raises(cash_magics, mock_shell):
    with _as_a_plain_cell("y: Undefined = 3"):
        run_cash_cell(cash_magics, "y: Undefined = 3")
    assert mock_shell.user_ns["y"] == 3, "the value is bound before the annotation is evaluated, as in plain Python"


def test_a_function_annotation_with_an_unknown_name_raises(cash_magics, mock_shell):
    source = "def f(a: Nope):\n    return 1"
    with _as_a_plain_cell(source) as raised:
        run_cash_cell(cash_magics, source)
    assert ("f" in mock_shell.user_ns) == (raised is None)


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
