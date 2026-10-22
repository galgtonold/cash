"""A pyplot draw does not name IPython's output history.

``Out`` keeps every value a cell showed. A cell ending in ``sns.barplot(...)``
puts the axes there, and with a backend that keeps figures open
(``MPLBACKEND=Agg``) it stays the current axes. Each later ``plt.xticks(...)``
named ``Out`` as drawn on, an input and output of the statement, and
digested every frame ever shown for its lineage: 9 s of a 0.4 s plotting
cell after twenty cells of a two-million-row log, 417 s after three hundred.

A method called on an axes ``Out`` holds (``Out[2].get_title()``,
``Out[2].set_title('x')``) named ``Out`` as the receiver it changes: it got a
lineage, and the next run evicted it as a name no cell binds, so the
statement raised ``NameError``. IPython's history is told by identity
(``Out is _oh``): a dict of the user's own named ``Out`` is still theirs.
"""

from __future__ import annotations

import ast

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from cash.notebook.pyplot_draws import pyplot_draw_names
from tests._cell_driver import run_cash_cell


@pytest.fixture
def axes():
    plt.close("all")
    _, ax = plt.subplots()
    yield ax
    plt.close("all")


def test_the_output_history_holding_the_current_axes_is_not_named(axes):
    out, inputs = {3: [1, 2], 5: axes}, ["", "sns.barplot()"]
    ns = {"plt": plt, "Out": out, "_oh": out, "In": inputs, "_ih": inputs, "ax": axes}
    assert pyplot_draw_names(ast.parse("plt.xticks(rotation=70)"), ns) == {"ax"}


def test_a_dict_of_the_users_own_holding_it_still_is(axes):
    ns = {"plt": plt, "Out": {5: axes}, "_oh": {}, "shown": {5: axes}}
    assert pyplot_draw_names(ast.parse("plt.tight_layout()"), ns) == {"Out", "shown"}


def test_a_method_called_on_an_axes_the_history_holds_leaves_it_bound(cash_magics, mock_shell):
    cells = [
        "import matplotlib.pyplot as plt",
        "fig, ax = plt.subplots()",
        "t = Out[2].get_title()",
        "Out[2].set_title('x')",
        "r = ax.get_title()",
    ]
    for cell in cells[:2]:
        run_cash_cell(cash_magics, cell, cells=cells)
    ns = mock_shell.user_ns
    ns["Out"] = ns["_oh"] = out = {2: ns["ax"]}
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert ns["Out"] is out
        assert "Out" not in cash_magics.tracking_state.variable_lineage
    assert out[2].get_title() == "x"
