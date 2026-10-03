"""Clean error display for notebook cell execution failures.

Builds synthetic tracebacks that point to the user's cell at the correct
line number, hiding internal cash framework frames.  This gives users the
same traceback experience they would get from native Jupyter execution.
"""

from __future__ import annotations

import ast
import linecache
import logging
import sys
import traceback as tb_mod
from typing import TYPE_CHECKING

from ..compiled_source import is_cash_filename

if TYPE_CHECKING:
    from cash.notebook._protocols import ShellProtocol

logger = logging.getLogger(__name__)


def show_clean_error(
    exc: Exception,
    raw_cell: str,
    node: ast.AST,
    shell: ShellProtocol,
) -> None:
    """Display *exc* with a clean traceback pointing to the user's cell.

    Instead of exposing cash framework internals (``_execute_cell``,
    ``__cash_exception__``), this builds a synthetic traceback whose
    top frame references ``Cell In[N]`` at the correct line — matching
    what Jupyter would show if the cell had been executed normally.

    Any deeper user-code frames (e.g. inside a function the user
    called) are preserved so the full call chain is visible.

    Parameters
    ----------
    exc:
        The exception raised during statement execution.
    raw_cell:
        Full source code of the notebook cell.
    node:
        AST node of the failing statement (used for ``lineno``).
    shell:
        IPython-compatible shell (needs ``execution_count`` and
        optionally ``showtraceback``).
    """
    exc_type = type(exc)

    # Determine the cell filename that IPython would use.
    try:
        exec_count = shell.execution_count
    except AttributeError:
        exec_count = 0
    cell_name = f"Cell In[{exec_count}]"

    # Register the full cell source in linecache so the traceback
    # formatter can display the relevant source lines.
    linecache.cache[cell_name] = (
        len(raw_cell),
        None,
        raw_cell.splitlines(True),
        cell_name,
    )

    # Determine the line number in the original cell for the failing
    # statement.  For control structures (for/if/while/with/try),
    # the error may come from a body statement deep inside.  In that
    # case, _cash_error_lineno is set by ControlStructureProcessor to
    # the exact body statement line.
    stmt_lineno: int = getattr(exc, "_cash_error_lineno", None) or getattr(node, "lineno", 1)

    # Build a synthetic cell-level traceback by compiling + exec'ing
    # a ``raise`` at the right line inside a code object whose
    # filename is ``Cell In[N]``.  IPython's traceback machinery will
    # then pick up the source from linecache and format it nicely.
    #
    # We also graft any deeper user-code frames (e.g. inside a
    # function the user called) from the original traceback so the
    # full call chain is visible.
    original_tb = exc.__traceback__
    user_inner_tb = None
    if original_tb is not None:
        tb = original_tb
        while tb is not None:
            if is_cash_filename(tb.tb_frame.f_code.co_filename):
                user_inner_tb = tb.tb_next  # frames below the cash-compiled unit
                break
            tb = tb.tb_next

    padded_code = "\n" * (stmt_lineno - 1) + "raise __cash_exc__"
    cell_code_obj = compile(padded_code, cell_name, "exec")

    try:
        exec(cell_code_obj, {"__cash_exc__": exc.with_traceback(user_inner_tb)})
    except BaseException:  # noqa: BLE001 - must catch all to synthesise the traceback
        _et, _ev, synth_tb = sys.exc_info()
        # synth_tb has two frames: this function's exec() → cell code.
        # We want only the cell code frame (and anything below it).
        cell_tb = synth_tb.tb_next or synth_tb

        # Use IPython's showtraceback if available (real notebook),
        # otherwise fall back to Python's traceback module (tests).
        if hasattr(shell, "showtraceback"):
            try:
                shell.showtraceback(
                    exc_tuple=(exc_type, exc, cell_tb),
                    tb_offset=0,
                    running_compiled_code=True,
                )
                return
            except (TypeError, AttributeError, ValueError):
                logger.debug("showtraceback call failed, falling back to print_exception")

        # Fallback: format and print directly.
        tb_mod.print_exception(exc_type, exc, cell_tb)


def show_module_load_error(module_name: str, exc: BaseException, shell: ShellProtocol) -> None:
    """Display why the edited module *module_name* did not reload.

    Shows what ``import module_name`` in a fresh kernel would show: a
    ``SyntaxError`` as its file, line and caret, any other error as a
    traceback that starts in the module's own file (the import machinery
    and cash's frames above it are dropped). One line first says the cell
    did not run.
    """
    module = sys.modules.get(module_name)
    path = getattr(module, "__file__", None) or module_name
    print(
        f"cash: {path} changed and no longer loads, so the kernel still holds its old code. "
        f"This cell did not run; fix the file and run it again.",
        file=sys.stderr,
    )
    tb = exc.__traceback__
    own = tb
    while own is not None and own.tb_frame.f_code.co_filename != path:
        own = own.tb_next
    tb = own if own is not None else tb
    try:
        # IPython shows a SyntaxError from the exception being handled, not
        # from ``exc_tuple``; raising it makes it that exception.
        raise exc.with_traceback(tb)
    except BaseException:  # noqa: BLE001 - only to make *exc* the handled exception
        if hasattr(shell, "showtraceback"):
            try:
                shell.showtraceback(exc_tuple=(type(exc), exc, tb), tb_offset=0)
                return
            except (TypeError, AttributeError, ValueError):
                logger.debug("showtraceback call failed, falling back to print_exception")
        tb_mod.print_exception(type(exc), exc, tb)
