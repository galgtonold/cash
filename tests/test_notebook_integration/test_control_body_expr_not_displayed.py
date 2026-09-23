"""A bare expression inside a for/if/try body is not displayed -- as in Jupyter.

A chart cell labelling bars with ``ax.text(...)`` in a
nested loop showed 151 ``Text(0, 0, '2582')`` reprs above the table. Jupyter
displays only a cell's LAST top-level expression; cash processes each body
statement on its own and treated every one as the last.
"""

import pytest

pytestmark = [pytest.mark.integration]

SETUP = (
    "class T:\n    def __init__(self, i): self.i = i\n"
    "    def __repr__(self): return f'T({self.i})'\n"
    "def f(i):\n    return T(i)"
)
BODIES = {
    "for": "for i in range(4):\n    f(i)\nprint('done')",
    "if_in_for": "for i in range(4):\n    if i % 2:\n        f(i)\nprint('done')",
    "nested_for_last": "for i in range(2):\n    for j in range(2):\n        f(i + j)",
    "if": "if True:\n    f(7)\nprint('done')",
    "try": "try:\n    f(8)\nexcept ValueError:\n    f(9)\nprint('done')",
}


def _shown(runner, cell):
    return [
        o.get("data", {}).get("text/plain")
        for o in runner.nb.cells[cell - 1].get("outputs", [])
        if o.get("output_type") in ("execute_result", "display_data")
        and "T(" in str(o.get("data", {}).get("text/plain", ""))
    ]


@pytest.mark.parametrize("body", list(BODIES))
def test_body_expression_is_not_displayed(nb_runner, body):
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, BODIES[body]])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _shown(nb_runner, 3) == []
    nb_runner.run_cell(3)  # warm: nothing replayed either
    assert _shown(nb_runner, 3) == []


def test_the_last_top_level_expression_still_is(nb_runner):
    nb_runner.create_notebook(["import cash\n%cash_on", SETUP, "for i in range(2):\n    f(i)\nf(5)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _shown(nb_runner, 3) == ["T(5)"]
