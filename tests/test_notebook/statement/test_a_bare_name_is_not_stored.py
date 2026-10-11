"""A statement that is only a name gets no entry.

``data`` alone costs its display, which a hit shows again. Stored, its value
was copied and pickled a second time: 0.6 s for a list of 87,000 sessions
whose display took 0.3 s, then refused as not worth its bytes.
"""

from __future__ import annotations

from cash.notebook.statement.store import StatementStore
from tests._cell_driver import run_cash_cell


def test_a_bare_name_with_a_slow_display_writes_nothing(cash_magics, monkeypatch):
    run_cash_cell(
        cash_magics,
        "import time\nclass Slow:\n    def __repr__(self):\n        time.sleep(0.05)\n        return 'slow'\nslow = Slow()",
    )
    written = []
    real = StatementStore._write

    def counting(self, run, *args, **kwargs):
        written.append(run.code)
        return real(self, run, *args, **kwargs)

    monkeypatch.setattr(StatementStore, "_write", counting)

    run_cash_cell(cash_magics, "slow")
    run_cash_cell(cash_magics, "x = (time.sleep(0.05), 1)[1]")

    assert written == ["x = (time.sleep(0.05), 1)[1]"]
