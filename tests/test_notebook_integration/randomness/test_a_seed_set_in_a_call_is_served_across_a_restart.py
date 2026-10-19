"""A call that seeds a global stream, or draws from one unseen, is served
after a kernel restart, and the next draw is the run's.

``s = setup()`` where ``setup`` seeds ``np.random`` or ``random`` (or draws
from one a cell above seeded) is stored once its draw is known, and a hit
puts the stream where the run left it and adopts its seed (344aa95). Across
real restarts, the next draw then matches the first run's, both for a cheap
draw and for one that is itself stored, which keys it as after the run.

The seed cell above the call is marked ``# @cash:persist`` so that it is
stored however fast it ran. Which session first serves the call or the cheap
draw can shift with machine load (the first run may only learn of a hidden
draw), so the test holds the values to the run's and asks only that the last
session serves the call, and the stored draw too.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

STATUS = (
    "[str(r['status']) for r in get_ipython().magics_manager.registry['CashMagics']._last_cell_metrics['statements']]"
)

#: case -> (cell above the call, the helper's body, what the next draw reads)
CASES = {
    "numpy seeded in a call": ("pass", ["np.random.seed(3)", "np.random.rand(4)"], "float(np.random.rand())"),
    "random seeded in a call": ("pass", ["random.seed(3)", "random.random()"], "random.random()"),
    "both seeded in a call": (
        "pass",
        ["random.seed(3)", "np.random.seed(4)", "np.random.rand(2)", "random.random()"],
        "(float(np.random.rand()), random.random())",
    ),
    "a hidden numpy draw after a seed above": (
        "np.random.seed(5)  # @cash:persist",
        ["np.random.rand(3)"],
        "float(np.random.rand())",
    ),
    "a hidden random draw after a seed above": (
        "random.seed(5)  # @cash:persist",
        ["random.random()"],
        "random.random()",
    ),
}


def _cells(case: str, stored_draw: bool) -> list[str]:
    above, body, draw = CASES[case]
    setup = "def setup():\n    time.sleep(0.4)\n" + "".join(f"    {line}\n" for line in body) + "    return 1"
    return [
        "import cash\n%cash_on",
        "import time, random\nimport numpy as np",
        above,
        setup,
        "s = setup()",
        f"x = (time.sleep(0.4), {draw})[1]" if stored_draw else f"x = {draw}",
    ]


@pytest.mark.parametrize("stored_draw", [False, True], ids=["a cheap draw", "a stored draw"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_the_next_draw_after_a_restart_is_the_runs(nb_runner, case, stored_draw):
    cells = _cells(case, stored_draw)
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    sessions = []
    for session in range(4):
        if session:
            nb_runner.restart()
        for cell in range(1, len(cells) + 1):
            nb_runner.run_cell(cell)
            if cell == 5:
                call = nb_runner.peek(STATUS)
        draw_status = nb_runner.peek(STATUS)
        nb_runner.settle_writes()
        sessions.append((call, draw_status, nb_runner.peek("x")))

    assert "RESTORED" in sessions[-1][0], sessions
    if stored_draw:
        assert "RESTORED" in sessions[-1][1], sessions
    assert len({x for _, _, x in sessions}) == 1, sessions
