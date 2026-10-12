"""A for loop whose target is an item or an attribute binds it, as Python does.

``for d['k'] in range(3)``, ``for ns.i in ...`` and ``for best['lr'] in
[0.1, 0.01]`` left the item unbound under cash: ``d`` stayed ``{}`` and the
body read the old item, on the very first run.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on",
    "d = {}\nfor d['k'] in range(3):\n    pass\nprint(d)",
    "import types\nns = types.SimpleNamespace()\nfor ns.i in range(3):\n    pass\nprint(ns)",
    "acc = [0, 0]\nfor acc[0] in [5, 6]:\n    acc[1] += acc[0]\nprint(acc)",
    "best = {'lr': None}\nscores = []\nfor best['lr'] in [0.1, 0.01]:\n    scores.append(best['lr'] * 10)\nprint(scores)",
]


def test_the_item_and_the_attribute_are_bound_each_pass(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "{'k': 2}" in nb_runner.get_output(2)
    assert "namespace(i=2)" in nb_runner.get_output(3)
    assert "[6, 11]" in nb_runner.get_output(4)
    assert "[1.0, 0.1]" in nb_runner.get_output(5)
