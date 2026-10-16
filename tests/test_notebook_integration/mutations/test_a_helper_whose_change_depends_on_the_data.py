"""A helper whose change to its argument depends on the data is watched again.

``def push_all(h, xs): for x in xs: heapq.heappush(h, x)`` changes ``h`` only
when ``xs`` has items. Its first run here has none, so cash sees ``h``
unchanged. When ``xs`` later has items, the change must still be seen: the
cell reading ``h`` must show the items, not what it read before. The same for
``for x in xs: push(h, x)``, a helper that always changes ``h`` called in a
loop that first ran on no items.
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration]

SLEEP = f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"

HELPERS = {
    "loop": "def push_all(h, xs):\n    for x in xs:\n        heapq.heappush(h, x)\n",
    "branch": "def push_all(h, xs):\n    if xs:\n        heapq.heappush(h, max(xs))\n",
    "early_return": "def push_all(h, xs):\n    if not xs:\n        return\n    heapq.heappush(h, max(xs))\n",
    "nested_helper": (
        "def _push(h, x):\n    heapq.heappush(h, x)\n\ndef push_all(h, xs):\n    for x in xs:\n        _push(h, x)\n"
    ),
}


@pytest.mark.parametrize(
    "call",
    ["push_all(h, xs)", "n = push_all(h, xs)", "for x in xs:\n    n = push_all(h, [x])"],
    ids=["bare", "kept", "in_a_loop"],
)
@pytest.mark.parametrize("helper", sorted(HELPERS))
def test_a_change_the_first_run_did_not_make_is_seen_later(nb_runner, helper, call):
    cells = [
        "import cash\n%cash_on",
        f"import heapq\nimport time\n{HELPERS[helper]}",
        "xs = []",
        f"h = []\n{call}",
        f"r = ({SLEEP}, sorted(h))[1]\nprint('H', r)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "H []" in nb_runner.get_output(5), nb_runner.get_output(5)

    nb_runner.set_cell_source(3, "xs = [5, 3, 9]")
    nb_runner.run_all()
    every_item = helper in ("loop", "nested_helper") or call.startswith("for")
    expected = "H [3, 5, 9]" if every_item else "H [9]"
    assert expected in nb_runner.get_output(5), nb_runner.get_output(5)
