"""A pyplot draw does not name IPython's output history.

``Out`` keeps every value a cell showed. A cell ending in ``sns.barplot(...)``
puts the axes there, and with a backend that keeps figures open
(``MPLBACKEND=Agg``) it stays the current axes. Each later ``plt.xticks(...)``
named ``Out`` as drawn on, an input and output of the statement, and
digested every frame ever shown for its lineage: 9 s of a 0.4 s plotting
cell after twenty cells of a two-million-row log, 417 s after three hundred.
"""

from __future__ import annotations

import ast

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
plt = pytest.importorskip("matplotlib.pyplot")

from cash.notebook.pyplot_draws import pyplot_draw_names


@pytest.fixture
def axes():
    plt.close("all")
    _, ax = plt.subplots()
    yield ax
    plt.close("all")


def test_the_output_history_holding_the_current_axes_is_not_named(axes):
    ns = {"plt": plt, "Out": {3: [1, 2], 5: axes}, "In": ["", "sns.barplot()"], "ax": axes}
    assert pyplot_draw_names(ast.parse("plt.xticks(rotation=70)"), ns) == {"ax"}


def test_a_dict_of_the_users_own_holding_it_still_is(axes):
    ns = {"plt": plt, "Out": {5: axes}, "shown": {5: axes}}
    assert pyplot_draw_names(ast.parse("plt.tight_layout()"), ns) == {"shown"}
