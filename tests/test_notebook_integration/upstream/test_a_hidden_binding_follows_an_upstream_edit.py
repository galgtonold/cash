"""A name bound by a header walrus, by ``exec`` or under ``global`` in a
function follows an edit above it, on an isolated run of a cell below.

cash re-ran ``v = base + 1`` but not what binds ``w``, leaving ``(6, 11)``:
neither the plain kernel's ``(6, 4)`` nor the top-to-bottom ``(20, 11)``.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

BINDERS = {
    "walrus": "if (w := base * 2) > 0:\n    pass",
    "exec": "exec('w = base * 2')",
    "global": "def setw():\n    global w\n    w = base * 2\nsetw()",
}


@pytest.mark.parametrize("binder", list(BINDERS.values()), ids=list(BINDERS))
def test_the_binding_reruns_with_its_neighbours(nb_runner, binder):
    nb_runner.create_notebook(["base = 3", binder, "v = base + 1", "y = (w, v)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(1, "base = 10")
    nb_runner.run_cell(4)

    assert nb_runner.peek("y") == "(20, 11)"
