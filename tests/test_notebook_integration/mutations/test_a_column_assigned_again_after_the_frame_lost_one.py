"""A column assigned again after the frame was reloaded and lost a column.

``df['tokens'] = ...`` runs on the loaded frame; the next cell reloads it
(``df = load()``) and takes a column out (``label = df.pop('label')``,
``del df['label']``); then the first cell runs again. It must add its column
to the frame as it is now, as plain Python does. Such a change did not move
``df``'s lineage, or was taken back on the re-run as the cell's own earlier
run, so the assignment was served its first entry: the whole frame of the
first run, ``label`` and all (held-out finding HO-1).
"""

import pytest

pytest.importorskip("pandas")

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\n%cash_badge print"
LOAD = (
    "import pandas as pd\n"
    "def load():\n"
    "    return pd.DataFrame({'text': ['a b', 'c d', 'e f'], 'label': [1, 2, 3]})\n"
    "df = load()"
)
# ``persist``: a statement this fast is otherwise not kept at all.
ASSIGN = "# @cash:persist\ndf['tokens'] = df.text.apply(str.split)"

CHANGES = {
    "pop_kept": "label = df.pop('label')",
    "pop_printed": "print(df.pop('label').sum())",
    "pop_bare": "df.pop('label')",
    "del": "del df['label']",
    "drop_in_place": "df.drop(columns=['label'], inplace=True)",
    "rename_in_place": "df.rename(columns={'label': 'y'}, inplace=True)",
}


def _plain(change: str) -> str:
    ns: dict = {}
    for code in (LOAD, ASSIGN, "df = load()", change, ASSIGN):
        exec(code, ns)
    return f"P {sorted(ns['df'].columns)} {ns['df'].shape}"


@pytest.mark.parametrize("case", sorted(CHANGES))
def test_the_assignment_lands_in_the_frame_as_it_is_now(nb_runner, case):
    change = CHANGES[case]
    # The second assignment is the same cell run again, after the cell that
    # reloads the frame and changes it.
    nb_runner.create_notebook([SETUP, LOAD, ASSIGN, f"df = load()\n{change}"])
    nb_runner.start_kernel()
    expected = _plain(change)
    for rep in range(2):
        if rep:
            nb_runner.settle_writes()
            nb_runner.restart()
            nb_runner._init_cash()
        nb_runner.run_cells([1, 2, 3, 4, 3])
        assert nb_runner.peek("f'P {sorted(df.columns)} {df.shape}'") == repr(expected), (
            f"run {rep}: {nb_runner.get_output(3)!r}"
        )
