"""A helper that sets the stream back to a saved state draws from where the
stream stood, not from a seed it set on the way (the temp-seed idiom:
``getstate``, ``seed(123)``, ``setstate``, draw). The draw follows the seed
set above it: after that seed is edited, Run All draws again rather than
serve what the old seed gave.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

SETUP = "import random, time\nimport numpy as np"
# The sleep makes it cost enough to be stored.
HELPERS = {
    "random, a temporary seed": (
        "random.seed({seed})",
        "def sample():\n    time.sleep(0.3)\n    st = random.getstate(); random.seed(123); random.setstate(st)\n"
        "    return random.random()",
    ),
    "numpy, a temporary seed": (
        "np.random.seed({seed})",
        "def sample():\n    time.sleep(0.3)\n    st = np.random.get_state(); np.random.seed(123); np.random.set_state(st)\n"
        "    return float(np.random.random())",
    ),
    "a draw, set back, then a seed": (
        "random.seed({seed})",
        "def sample():\n    time.sleep(0.3)\n    st = random.getstate(); v = random.random(); random.setstate(st)\n"
        "    random.seed(5)\n    return v",
    ),
    "a draw under the temporary seed, then from the stream": (
        "random.seed({seed})",
        "def sample():\n    time.sleep(0.3)\n    st = random.getstate(); random.seed(123); a = random.random()\n"
        "    random.setstate(st)\n    return a + random.random()",
    ),
}


def _plain(seed_cell: str, helper: str) -> str:
    namespace: dict = {}
    exec(f"{SETUP}\n{seed_cell}\n{helper.replace('time.sleep(0.3)', 'pass')}\nv = sample()", namespace)
    return f"M {namespace['v']}"


@pytest.mark.parametrize("how", list(HELPERS))
def test_an_edited_seed_reaches_the_draw(nb_runner, how):
    pytest.importorskip("numpy")
    seed, helper = HELPERS[how]
    nb_runner.create_notebook([SETUP, seed.format(seed=0), helper, "v = sample()", "print('M', v)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _plain(seed.format(seed=0), helper) in nb_runner.get_raw_output(5)

    nb_runner.set_cell_source(2, seed.format(seed=1))
    nb_runner.run_all()
    shown = nb_runner.get_raw_output(5)
    assert _plain(seed.format(seed=1), helper) in shown, shown
