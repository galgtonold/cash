"""A class a library function makes in a cell names the cell's module.

``collections.namedtuple`` names the module of the frame that called it.
Called through the call cache, that was cash's own frame: ``Pipe`` named
``cash.notebook.call_effects``, so its objects could not be pickled by
reference, and cash took the class for a library's (a namedtuple of
functions handed to a helper was not looked into).
"""

from __future__ import annotations

import warnings

from tests._cell_driver import run_cash_cell


def test_a_namedtuple_made_in_a_cell_is_the_cells(cash_magics, mock_shell):
    code = "import collections\nPipe = collections.namedtuple('Pipe', 'pre post')\nrow = Pipe(1, 2)"
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_cash_cell(cash_magics, code, cells=[code])
    ns = mock_shell.user_ns
    assert ns["Pipe"].__module__ == ns.get("__name__", "__main__")
    assert ns["row"] == (1, 2)


def test_an_error_in_the_library_call_still_reaches_the_cell(cash_magics, mock_shell):
    """Control: an exception raised by the library function is the cell's,
    as before."""
    code = "import collections\nPipe = collections.namedtuple('Pipe', 'pre 1post')"
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            run_cash_cell(cash_magics, code, cells=[code])
        except ValueError:
            pass
    assert "Pipe" not in mock_shell.user_ns
