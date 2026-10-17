"""A display a statement sent is shown again exactly as it was sent.

cash runs a statement under IPython's ``capture_output`` and shows what it
captured afterwards. It passed each captured ``RichOutput`` to ``display()``,
which formats it as an object: a bundle with no ``text/plain`` (a plotly
figure's, or any ``_ipython_display_`` that sends a raw bundle) got the
object's repr there, so under the figure the notebook showed
``<IPython.utils.capture.RichOutput at 0x..>``. Plain showed nothing. It
happened on the run that computes the statement and on a run that serves it.
"""

from __future__ import annotations

import ast

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

_STATUSES = (
    "[str(s.get('status')) for s in "
    "get_ipython().magics_manager.magics['line']['cash_status'].__self__"
    "._last_cell_metrics['statements']]"
)

THING = (
    "import time\n"
    "from IPython.display import display\n"
    "class Thing:\n"
    "    def _ipython_display_(self):\n"
    "        display({'application/x-thing+json': {'n': 1}, 'text/html': '<b>thing</b>'}, raw=True)\n"
    "def shown(n):\n"
    "    time.sleep(0.4)  # slow enough to store\n"
    "    display(Thing())\n"
    "    return sum(range(n))"
)


def _bundles(nb_runner, cell_num):
    cell = nb_runner.nb.cells[cell_num - 1]
    return [dict(o.get("data", {})) for o in cell.get("outputs", []) if o.output_type == "display_data"]


def _statuses(nb_runner):
    return ast.literal_eval(nb_runner.peek(_STATUSES))


def _assert_shown_as_sent(nb_runner, cell_num, mime):
    bundles = _bundles(nb_runner, cell_num)
    assert len(bundles) == 1, bundles
    assert mime in bundles[0], bundles
    assert "RichOutput" not in bundles[0].get("text/plain", ""), bundles


def test_a_raw_bundle_from_ipython_display_is_shown_as_sent(nb_runner):
    # The badge is a display too; off, a cell's displays are the user's own.
    nb_runner.create_notebook(["%cash_badge off", THING, "display(Thing())", "total = shown(3)"])
    nb_runner.start_kernel()

    nb_runner.run_all()
    assert _statuses(nb_runner) == ["COMPUTED"]
    for cell in (3, 4):
        _assert_shown_as_sent(nb_runner, cell, "application/x-thing+json")
        assert "text/plain" not in _bundles(nb_runner, cell)[0]

    nb_runner.run_all()
    assert _statuses(nb_runner) == ["RESTORED"]  # served: its display is replayed from the cache
    for cell in (3, 4):
        _assert_shown_as_sent(nb_runner, cell, "application/x-thing+json")
        assert "text/plain" not in _bundles(nb_runner, cell)[0]


def test_a_plotly_figure_shows_no_stray_line(nb_runner):
    pytest.importorskip("plotly")
    nb_runner.create_notebook(
        [
            "%cash_badge off\nimport plotly.express as px",
            "fig = px.scatter(x=[1, 2, 3], y=[1, 4, 9])\nfig.show()",
        ]
    )
    nb_runner.start_kernel()

    nb_runner.run_all()
    assert _statuses(nb_runner) == ["COMPUTED", "COMPUTED"]
    _assert_shown_as_sent(nb_runner, 2, "application/vnd.plotly.v1+json")

    nb_runner.run_all()
    assert _statuses(nb_runner) == ["RESTORED", "RESTORED"]
    _assert_shown_as_sent(nb_runner, 2, "application/vnd.plotly.v1+json")
