"""A call that displays something shows it on every run, as in plain.

A cached call puts back what it printed on a hit, not what it displayed. So a
statement too quick to store, re-run around a stored call (``m = rows(2)``,
with ``rows`` printing and then ``display``-ing a table), showed the prints
and no table. Such a call is not stored: it runs, and shows both.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

HELPERS = (
    "import cash\n"
    "from IPython.display import display, HTML\n"
    "def rows(n):\n"
    "    for i in range(n):\n"
    "        print('row', i)\n"
    "    display(HTML('<b>shown</b>'))\n"
    "    return n\n"
    # The call is worth storing, the statement around it is not.
    "cash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=5.0)\n"
)


def _outputs(nb_runner, cell_num):
    seen = []
    for o in nb_runner.nb.cells[cell_num - 1].outputs:
        if o.output_type == "stream" and o.name == "stdout":
            seen.extend(line for line in o.text.splitlines() if line and not line.startswith("[Cash]"))
        elif o.output_type == "display_data" and "<b>shown</b>" in o.data.get("text/html", ""):
            seen.append("html")
    return seen


def test_a_call_that_displays_shows_it_when_the_statement_runs_again(nb_runner):
    nb_runner.create_notebook([HELPERS, "m = rows(2)"])
    nb_runner.start_kernel()
    nb_runner.run_cell(1)

    for _ in range(2):
        nb_runner.run_cell(2)
        assert _outputs(nb_runner, 2) == ["row 0", "row 1", "html"]
