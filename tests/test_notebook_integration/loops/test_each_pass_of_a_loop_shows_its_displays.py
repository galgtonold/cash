"""Every pass of a loop shows its own displays, after its own printed output.

cash runs a loop pass by pass and showed a pass's prints as it finished, but
held its displays back as the cell's result, and each pass's replaced the
last: ``display()`` in a loop showed only the last item, after all the prints.
Plain shows ``p 0``, ``item 0``, ``p 1``, ``item 1``, ...
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

SHOW = (
    "import time\n"
    "from IPython.display import display, Markdown\n"
    "def show(n):\n"
    "    time.sleep(0.3)  # slow enough to store\n"
    "    print('p', n)\n"
    "    display(Markdown(f'item {n}'))"
)


def _shown(nb_runner, cell_num):
    shown = []
    for o in nb_runner.nb.cells[cell_num - 1].get("outputs", []):
        if o.output_type == "stream":
            shown += [line for line in o.text.splitlines() if line]
        elif o.output_type == "display_data":
            shown.append(o.data.get("text/markdown"))
    return shown


def test_a_display_in_a_loop_shows_every_item_in_order(nb_runner):
    nb_runner.create_notebook(["%cash_badge off", SHOW, "for n in range(3):\n    show(n)"])
    nb_runner.start_kernel()
    plain = ["p 0", "item 0", "p 1", "item 1", "p 2", "item 2"]

    nb_runner.run_all()
    assert _shown(nb_runner, 3) == plain
    assert "COMPUTED" in ast.literal_eval(nb_runner.peek(_STATUSES))

    nb_runner.run_all()
    statuses = ast.literal_eval(nb_runner.peek(_STATUSES))
    assert "COMPUTED" not in statuses, statuses  # served from the cache
    assert _shown(nb_runner, 3) == plain
