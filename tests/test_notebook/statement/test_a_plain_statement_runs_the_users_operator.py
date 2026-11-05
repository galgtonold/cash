"""``z = a + 1`` is a plain statement (``plain_statement``: no call in its
text), yet ``+`` runs the user's ``__add__``. The analyses that skip a
plain statement only look for calls, attributes and subscripts; what the
operator runs is still read, so a re-run runs it again, as plain Python
does."""

from __future__ import annotations

from tests._cell_driver import run_cash_cell

SETUP = "LOG = []\nclass Adds:\n    def __add__(self, other):\n        LOG.append(other)\n        return 0\na = Adds()"


def test_the_operator_runs_on_each_run(cash_magics, mock_shell):
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, "z = a + 1")
    run_cash_cell(cash_magics, "z = a + 1")
    assert mock_shell.user_ns["LOG"] == [1, 1]
    assert mock_shell.user_ns["z"] == 0
