"""A draw cash runs again above a cell is reseeded the way a run from the top
seeds it, also when the seed sits in a notebook helper (``set_seed(0)``) or
the draw sits inside a library call (``df.sample``): the numbers are a run
from the top's, and nothing warns.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream, pytest.mark.timeout(120)]

FORMS = {
    "seed in a helper": (
        "import numpy as np, time",
        "def set_seed(s):\n    np.random.seed(s)\nset_seed(0)",
        "m = [k * x for x in np.random.rand(2)] + [time.sleep(0.2)]",
    ),
    "draw in a library call": (
        "import numpy as np, pandas as pd, time",
        "np.random.seed(0)\ndf = pd.DataFrame({'x': range(50)})",
        "m = list(k * df.sample(2)['x']) + [time.sleep(0.2)]",
    ),
}


@pytest.mark.parametrize("how", list(FORMS))
def test_a_redrawn_value_is_drawn_from_the_seed_above(nb_runner, how):
    pytest.importorskip("pandas")
    imports, seed, draw = FORMS[how]
    cells = ["%cash_badge print\n" + imports, seed, "k = 2", draw, "print('M', m[:2])"]
    namespace: dict = {}
    exec("\n".join([imports, seed, "k = 3", draw.replace("time.sleep(0.2)", "0")]), namespace)

    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "k = 3")
    nb_runner.run_cell(5)
    shown = nb_runner.get_raw_output(5)
    assert f"M {namespace['m'][:2]}" in shown, shown
    assert "RANDOM-REDRAWN" not in shown, shown
