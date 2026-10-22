"""A list the notebook holds that a frame holds in an object cell stays that
cell after a hit.

``rect = [1, 2]`` then ``df = pd.DataFrame({..., 'm': [..., rect, ...]})``:
the share check finds ``rect`` holding one of ``df``'s cells, but no patch
could name a frame's cell, so ``rect`` was stored whole with ``df``. The
tier copies a frame on its own, and the served ``df`` held a copy of the
list apart from the restored ``rect``: ``rect.append(3)`` no longer showed
in ``df``. The cell is now named by its column and row, and ``rect`` takes
the restored one back.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


SETUP = (
    "rect = [1, 2]\n"
    "class K:\n"
    "    def __init__(self, v):\n"
    "        self.v = v\n"
    "    def __hash__(self):\n"
    "        return hash(self.v)\n"
    "    def __eq__(self, o):\n"
    "        return isinstance(o, K) and o.v == self.v\n"
    "df = pd.DataFrame({'x': np.arange(-3, 7), 's': [f's{i}' for i in range(10)], "
    "'o': [{'i': i} for i in range(10)], 'l': [[i] for i in range(10)], "
    "'m': ['a', 1, 2.5, rect, None, 'b', 3, (1, 2), b'x', {'k': 1}], "
    "'c': pd.Categorical(['u', 'v'] * 5), 'ck': pd.Categorical([K(i % 3) for i in range(10)])})"
)


def test_the_served_frame_holds_the_list(nb_runner):
    cells = [
        "import cash\n%cash_on\nimport numpy as np, pandas as pd",
        SETUP,
        "rect.append(3)\nprint('R', df['m'].iloc[3] is rect, df['m'].iloc[3])",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for restart in (False, False, False, True):
        if restart:
            nb_runner.restart()
        for i in range(1, len(cells) + 1):
            nb_runner.run_cell(i)
        assert "R True [1, 2, 3]" in nb_runner.get_output(3), nb_runner.get_output(3)
