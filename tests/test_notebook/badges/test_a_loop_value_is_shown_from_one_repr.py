"""The badge shows each loop value from one ``repr`` per render.

Each iteration's values appear in every body statement's histogram tooltip
and drill-down row, and in the loop's header. On E11 the item of a 6-image
loop, a tuple holding a tensor, was repr'd hundreds of times a run, 0.4-0.8 s
next to a cell of 1.5 s (`_binding_repr`).
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell

ITEMS = (
    "calls = []\n"
    "class Item:\n"
    "    def __init__(self, k):\n"
    "        self.k = k\n"
    "    def __repr__(self):\n"
    "        calls.append(self.k)\n"
    "        return f'Item({self.k})'\n"
    "items = [Item(k) for k in range(4)]"
)


def test_each_value_is_repr_d_once(cash_magics, mock_shell, monkeypatch):
    from cash.notebook.ipython.badges import BadgePresenter

    # One render: a cell running past the interval (a busy machine) draws a
    # progress badge too, which shows each value again.
    monkeypatch.setattr(BadgePresenter, "MIN_RENDER_INTERVAL", 60.0)
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(cash_magics, ITEMS)
    run_cash_cell(cash_magics, "for item in items:\n    a = item.k\n    b = a + 1\n    c = b * 2")
    calls = mock_shell.user_ns["calls"]
    assert sorted(calls) == [0, 1, 2, 3]
