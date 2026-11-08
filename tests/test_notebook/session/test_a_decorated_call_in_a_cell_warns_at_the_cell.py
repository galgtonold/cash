"""A warning about a ``@cash.cache`` function called from a notebook cell
names the line in the cell as cash's notebook warnings do (``<cash>:N``),
not the pseudo-filename cash compiled the cell's statement under
(``/<cash-1a2b>``), which the reader sees nowhere else."""

from __future__ import annotations

import warnings

from tests._cell_driver import run_cash_cell


def test_the_warning_names_the_line_in_the_cell(cash_magics, cash_instance, mock_shell):
    mock_shell.user_ns["cash_instance"] = cash_instance
    run_cash_cell(
        cash_magics,
        "class Box:\n    pass\n\n@cash_instance.cache\ndef fill(box):\n    vars(box).update(n=1)\n    return 1\n",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, "n = 1\nfill(Box())\n")
    found = [w for w in caught if getattr(w.message, "code", None) == "IMPURE-OBSERVED-EFFECTS"]
    assert [(w.filename, w.lineno) for w in found] == [("<cash>", 2)], [str(w.message) for w in caught]


def test_a_filter_on_the_notebooks_module_still_reaches_it(cash_magics, cash_instance, mock_shell):
    """Only the name shown changed: a cell runs as ``__main__``, and a
    filter naming that module quiets the warning as it does any other."""
    mock_shell.user_ns["cash_instance"] = cash_instance
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(
        cash_magics,
        "class Box:\n    pass\n\n@cash_instance.cache\ndef fill(box):\n    vars(box).update(n=1)\n    return 1\n",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings("ignore", module="__main__")
        run_cash_cell(cash_magics, "n = 1\nfill(Box())\n")
    assert not [w for w in caught if getattr(w.message, "code", None) == "IMPURE-OBSERVED-EFFECTS"]


def test_a_notebook_warning_with_no_cell_to_name_shows_no_line_zero():
    from cash.diagnostics import warn_diagnostic
    from cash.notebook.server_discovery import CashNotebookDiscoveryWarning

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_diagnostic(CashNotebookDiscoveryWarning, "NOTEBOOK-NOT-FOUND", "a test", "nothing", location=("<cash>", 0))
        warn_diagnostic(CashNotebookDiscoveryWarning, "NOTEBOOK-NOT-FOUND", "a test", "nothing", location=("<cash>", 3))
        warnings.filterwarnings("ignore", module="__main__")
        warn_diagnostic(CashNotebookDiscoveryWarning, "NOTEBOOK-NOT-FOUND", "a test", "nothing", location=("<cash>", 4))
    shown = [(w.filename, w.lineno) for w in caught]
    assert len(shown) == 2, shown
    assert all(lineno > 0 for _, lineno in shown), shown
    assert shown[1] == ("<cash>", 3), shown
