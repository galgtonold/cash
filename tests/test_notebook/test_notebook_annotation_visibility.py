"""`# @cash:assume-safe` must work for a function defined in a cell.

cash splits a cell with ``ast.unparse``, which strips comments, then compiles
that text and registers it in linecache. So ``inspect.getsource`` on a function
defined in a cell returned source with no comments in it, and the per-line
purity waiver -- read back from the function's own source -- was invisible.

Measured before the fix:

    in a module file      annotated 0 warnings, unannotated 1
    in a %cash_on cell    annotated 1 warning,  unannotated 1

Both arms are required. Without the unannotated arm, a run where warnings are
broken generally would look like a pass.
"""
from __future__ import annotations

import asyncio
import tempfile
import warnings

import pytest

pytest.importorskip("IPython")

from cash import Cash
from cash.notebook.ipython.cell_executor import _PipelineSyntaxError
from cash.notebook.ipython.magics import CashMagics
from tests.conftest import MockShell


@pytest.fixture
def cell_runner():
    """Run cells through the real magic, returning the impurity-warning count."""
    shell = MockShell()
    cash = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)
    magics = CashMagics(shell, cash)
    magics.cash_on("")
    magics._badge_mode = "off"          # keep the badge out of the captured output
    shell.user_ns["c"] = cash

    def run(cell: str) -> int:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            magics.cash("", cell)
        return len([w for w in caught if "Impurity" in type(w.message).__name__])

    return run


ANNOTATED = (
    "import time\n"
    "@c.cache\n"
    "def audited(n):\n"
    "    time.sleep(0.01)  # @cash:assume-safe - deliberate, this is the test\n"
    "    return n * 2\n"
    "audited(1)\n"
)

PLAIN = (
    "import time\n"
    "@c.cache\n"
    "def unaudited(n):\n"
    "    time.sleep(0.01)\n"
    "    return n * 2\n"
    "unaudited(1)\n"
)


def test_an_annotated_line_is_waived_in_a_notebook_cell(cell_runner):
    assert cell_runner(ANNOTATED) == 0, (
        "the waiver was not honoured for a function defined in a cell"
    )


def test_an_unannotated_line_still_warns_in_a_notebook_cell(cell_runner):
    """The control: proves the test above is not passing because warnings are off."""
    assert cell_runner(PLAIN) == 1


def test_a_pep614_parenthesised_decorator_does_not_kill_the_cell():
    """FINDING 1 (review round 1). A decorator whose expression begins on the
    line AFTER the ``@`` -- legal since PEP 614 (Python 3.9) -- made
    ``_exec_source_for_node`` prepend from ``decorator_list[0].lineno``,
    which is the EXPRESSION's line, not the ``@`` line. The recovered text
    started mid-expression (``'    c.cache\\n)\\ndef f(n):...'``): the ``@(``
    line dropped, a stray ``)`` left behind. That does not compile, and
    handing it to ``compile()`` in ``_execute_statement`` killed the cell
    with an ``IndentationError`` -- valid user code that ran fine on base
    ``479e30e``, so this was a genuine regression, not a pre-existing hole.

    The general fix -- ``compile()`` the recovered text before returning it,
    ``None`` if that raises -- falls back to the unparsed form here exactly
    as it does for any other unrecoverable statement (per
    ``_exec_source_for_node``'s own docstring promise: "this must never be
    able to break a cell"). The annotation is then simply not visible to
    ``inspect.getsource`` for this one rare shape -- the same behaviour
    every function had before Task 1 -- but the cell runs.
    """
    shell = MockShell()
    cash = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)
    magics = CashMagics(shell, cash)
    magics.cash_on("")
    magics._badge_mode = "off"
    shell.user_ns["c"] = cash

    cell = (
        "@(\n"
        "    c.cache\n"
        ")\n"
        "def f(n):\n"
        "    return n  # @cash:assume-safe\n"
        "f(3)\n"
    )

    magics.cash("", cell)  # must not raise -- this is the regression itself

    assert "f" in shell.user_ns, "the cell aborted before the def bound the name"
    assert shell.user_ns["f"](3) == 3, "the fallback-executed function must still work"


@pytest.fixture
def async_cell_runner():
    """Async twin of ``cell_runner``.

    ``%%cash`` (``magics.cash(...)``) is a purely SYNC entry point -- its
    split loop (``_execute_cell_statements``) only ever calls
    ``process_statement`` (never the ``_async`` twin), so it cannot exercise
    ``_execute_statement_async`` no matter what the cell contains. The
    ``_async`` path is only reached via ``execute_cell_async``, which in a
    real notebook is dispatched by IPython's own ``run_cell_async`` (the
    ``%cash_on`` monkeypatch) when a cell contains a top-level ``await`` --
    routing that needs a live kernel event loop to reproduce end-to-end.

    Calling ``execute_cell_async`` directly, the same way
    ``test_await_in_control_body.py`` calls
    ``process_await_unit`` directly, exercises the real
    ``process_statement_async`` / ``_execute_statement_async`` machinery
    without needing one.
    """
    shell = MockShell()
    cash = Cash(cache_dir=tempfile.mkdtemp(), register_magic=False)
    magics = CashMagics(shell, cash)
    magics.cash_on("")
    magics._badge_mode = "off"          # keep the badge out of the captured output
    shell.user_ns["c"] = cash

    async def _tick(n):
        await asyncio.sleep(0)
        return n

    shell.user_ns["_tick"] = _tick

    def run(cell: str) -> int:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = asyncio.run(magics._cell_executor.execute_cell_async(cell))
        assert not isinstance(result, _PipelineSyntaxError), (
            "a cell with a top-level await must not fail to parse on the async path"
        )
        return len([w for w in caught if "Impurity" in type(w.message).__name__])

    return run


ANNOTATED_ASYNC = (
    "import time\n"
    "@c.cache\n"
    "def audited(n):\n"
    "    time.sleep(0.01)  # @cash:assume-safe - deliberate, this is the test\n"
    "    return n * 2\n"
    "audited(1)\n"
    "_ = await _tick(1)\n"
)

PLAIN_ASYNC = (
    "import time\n"
    "@c.cache\n"
    "def unaudited(n):\n"
    "    time.sleep(0.01)\n"
    "    return n * 2\n"
    "unaudited(1)\n"
    "_ = await _tick(1)\n"
)


def test_an_annotated_line_is_waived_in_a_notebook_cell_with_top_level_await(async_cell_runner):
    """FINDING 6 (review round 1). The plumbing is symmetric -- the reviewer
    confirmed by runtime spy that ``exec_source`` reaches
    ``_execute_statement_async`` and that the CAS-243 guard fires there too
    -- but nothing in the suite pinned it, and this project has shipped
    one-sided sync/async fixes before. Async twin of
    ``test_an_annotated_line_is_waived_in_a_notebook_cell``: a cell with a
    top-level ``await`` alongside the same annotated cell-defined function
    must still waive the purity warning, going through ``execute_cell_async``
    / ``process_statement_async`` / ``_execute_statement_async`` instead of
    their sync twins."""
    assert async_cell_runner(ANNOTATED_ASYNC) == 0, (
        "the waiver was not honoured on the async execution path"
    )


def test_an_unannotated_line_still_warns_in_a_notebook_cell_with_top_level_await(async_cell_runner):
    """The control: proves the test above is not passing because warnings
    are off, or because the async path silently skips the purity check
    altogether."""
    assert async_cell_runner(PLAIN_ASYNC) == 1
