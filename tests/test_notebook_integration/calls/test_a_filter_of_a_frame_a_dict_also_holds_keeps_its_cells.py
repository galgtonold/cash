"""``df2 = flt(df)`` keeps ``df``'s cells when ``df`` is also in a dict.

``frames = {'a': df}`` reaches the outputs' objects only through ``df``,
whose cells the entry takes back by row. That holder could not be given
places of its own, so every holder was stored whole, and a hit handed back
``df2`` with copies of ``df``'s cells: a change made through ``df`` no longer
showed in ``df2``. The dict is now left as it is, like ``df``.
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SLEEP = ABOVE_PERSISTENCE_FLOOR_S * 2


@pytest.mark.parametrize("holder", ["{'a': df}", "[df]"])
@pytest.mark.parametrize("rows", [10, 200])
def test_the_served_filter_shares_the_frames_cells(nb_runner, holder, rows):
    cells = [
        "import cash\n%cash_on\nimport time\nimport numpy as np, pandas as pd",
        f"df = pd.DataFrame({{'x': np.arange(-3, {rows} - 3), 'o': [{{'i': i}} for i in range({rows})]}})\n"
        f"frames = {holder}",
        f"def flt(d):\n    time.sleep({SLEEP})\n    return d[d.x > 0]",
        "df2 = flt(df)",
        "df['o'].iloc[4]['z'] = 1\nprint('R', df2['o'].loc[4] is df['o'].loc[4], dict(df2['o'].loc[4]))",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for restart in (False, False, True):
        if restart:
            nb_runner.restart()
        nb_runner.run_all()
        assert "R True {'i': 4, 'z': 1}" in nb_runner.get_output(5), nb_runner.get_output(5)
