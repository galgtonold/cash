"""An output holding a frame's cells deeper than the two levels cash reads,
or past the bound of the second, may hold any frame's cells, and cash takes
it to: a change made in place through it moves every frame with an object
column, and a reader of ``df['o']`` below is not served what it read
before. ``H = {'a': {'L': list(df['o'])}}`` changed through
``H['a']['L'][3]['z']`` was served stale after the edit and after a
restart, and so was a list holding the cells after four million other
items.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mutations, pytest.mark.timeout(240)]

FRAME = "df = pd.DataFrame({'x': np.arange(100_000.), 'o': [{'i': i} for i in range(100_000)]})"
HOLDERS = {
    "dict_of_dict_of_list": ("H = {'a': {'L': list(df['o'])}}", "H['a']['L'][3]['z'] = {z}"),
    "tuple_of_dict_of_list": ("H = ({'L': list(df['o'])},)", "H[0]['L'][3]['z'] = {z}"),
    "past_the_bound": ("H = {'pad': [{}] * 4_000_000, 'L': list(df['o'])}", "H['L'][3]['z'] = {z}"),
}
READER = "g = (time.sleep(0.3), df['o'].map(str).tolist())[1]"


@pytest.mark.parametrize("holder", list(HOLDERS))
def test_the_reader_below_sees_the_change_after_an_edit_and_a_restart(nb_runner, holder):
    hold, edit = HOLDERS[holder]
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\nimport time\nimport numpy as np, pandas as pd",
            f"{FRAME}\n{hold}",
            edit.format(z=1),
            READER,
            "print('R', g[3], df['o'].map(str).tolist()[3])",
        ]
    )
    nb_runner.start_kernel()
    got = []
    for z, restart in ((1, False), (2, False), (2, True)):
        if restart:
            nb_runner.restart()
        nb_runner.set_cell_source(3, edit.format(z=z))
        nb_runner.run_all()
        got.append(nb_runner.get_output(5).strip())
    assert got == [f"R {{'i': 3, 'z': {z}}} {{'i': 3, 'z': {z}}}" for z in (1, 2, 2)], got
