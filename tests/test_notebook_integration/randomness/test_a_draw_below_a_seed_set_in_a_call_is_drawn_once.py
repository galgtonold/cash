"""A draw below a statement whose call seeds the stream is drawn once, as in
plain Python, on a second Run All too, and served after a restart.

``suite = (rc(5), rc(6)) * 4`` with ``rc`` seeding ``random`` (E16's
``random_cities``), then ``m = mean(... random.sample(t, 2) ...)``: the print
cell below saw ``m`` as stale with nothing changed, ran it again as an
upstream repair and printed a mean drawn a second time.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

SETUP = "import random, time\nfrom functools import cache\nfrom statistics import mean\nimport numpy as np"
# The sleep makes it cost enough to be stored.
DRAW = "m = mean((time.sleep(0.05) or min(random.sample(t, 2))) for t in suite)"
NP_DRAW = "m = mean((time.sleep(0.05) or int(np.random.choice(t, 2).min())) for t in suite)"

CASES = {
    "a function": ("def rc(n):\n    random.seed(n)\n    return tuple(range(n))", "suite = (rc(5), rc(6)) * 4", DRAW),
    "a memo that seeds, then draws": (
        "@cache\ndef rc(n):\n    random.seed(n)\n    return tuple(round(random.uniform(0, 9), 3) for _ in range(n))",
        "suite = (rc(5), rc(6)) * 4",
        DRAW,
    ),
    "a method": (
        "class Maker:\n    def rc(self, n):\n        random.seed(n)\n        return tuple(range(n))\nmaker = Maker()",
        "suite = (maker.rc(5), maker.rc(6)) * 4",
        DRAW,
    ),
    "a function map calls": (
        "def rc(n):\n    random.seed(n)\n    return tuple(range(n))",
        "suite = tuple(map(rc, [5, 6])) * 4",
        DRAW,
    ),
    "numpy's seed": (
        "def rc(n):\n    np.random.seed(n)\n    return tuple(range(n))",
        "suite = (rc(5), rc(6)) * 4",
        NP_DRAW,
    ),
}


@pytest.mark.parametrize("how", list(CASES))
def test_the_draw_is_made_once_and_served_after_a_restart(nb_runner, how):
    pytest.importorskip("numpy")
    helper, suite, draw = CASES[how]
    namespace: dict = {}
    exec(f"{SETUP}\n{helper}\n{suite}\n{draw.replace('time.sleep(0.05) or ', '')}", namespace)
    expected = f"M {namespace['m']}"

    nb_runner.create_notebook(["%cash_badge print\n" + SETUP, helper, suite, draw, "print('M', m)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    shown = nb_runner.get_raw_output(5)
    assert expected in shown and "Upstream" not in shown, shown
    nb_runner.run_all()
    assert expected in nb_runner.get_raw_output(5), nb_runner.get_raw_output(5)

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert "CACHED: m = mean" in nb_runner.get_raw_output(4), nb_runner.get_raw_output(4)
    assert expected in nb_runner.get_raw_output(5), nb_runner.get_raw_output(5)
