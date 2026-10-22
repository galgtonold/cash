"""Review repro: the check is one-directional (memo now holds >= what the
run left). An entry stored when the helper MISSED (its body ran, printed)
is served where the memo already holds the answer and plain Python would
hit silently: the served statement replays the miss's output.
Place under tests/test_notebook/restore/.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

IMPORTS = "import functools, time"
HELPER = "@functools.cache\ndef sq(n):\n    print('computing', n)\n    return n * n"
FILL = f"x = (sq(3), time.sleep({ABOVE_PERSISTENCE_FLOOR_S}))[0]"


def _restart(magics, names) -> None:
    magics.tracking_state.reset_session_state()
    magics._upstream_checker.simulator.cache.reset()
    for name in names:
        magics.shell.user_ns.pop(name, None)


def _statuses(magics) -> list[str]:
    return [str(m.get("status")) for m in magics.cash_status("dict")["last_cell"]["statements"]]


def test_entry_stored_on_a_miss_served_where_the_memo_hits(cash_magics, capsys):
    for code in [IMPORTS, HELPER, FILL]:
        run_cash_cell(cash_magics, code)
    _restart(cash_magics, ["functools", "time", "sq", "x", "_"])
    for code in [IMPORTS, HELPER, "_ = sq(3)"]:
        run_cash_cell(cash_magics, code)
    capsys.readouterr()
    run_cash_cell(cash_magics, FILL)
    out = capsys.readouterr().out
    # plain Python: sq(3) hits, nothing printed
    assert "computing" not in out, f"status={_statuses(cash_magics)} printed={out!r}"
