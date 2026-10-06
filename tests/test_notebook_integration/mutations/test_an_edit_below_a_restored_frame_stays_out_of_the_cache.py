"""An edit to a restored frame below the cell that built it never reaches the
cache: re-running the cell gives a fresh value, as in a plain kernel.

The RAM tier handed back frames that shared their data with the stored entry:
a write through ``.values`` of a nullable column, or into a list cell of a
frame held by a dataclass, survived re-running the cell that built it.
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.timeout(120)]


def test_a_write_through_values_does_not_survive_a_rerun(nb_runner):
    nb_runner.create_notebook(
        [
            "import time\nimport pandas as pd",
            f"time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
            "df = pd.DataFrame({'score': pd.array([1, 2, 3], dtype='Int64')})",
            "df['score'].values[0] = 100",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(2)
    assert nb_runner.peek("int(df['score'][0])").strip() == "1"


@pytest.mark.fresh_kernel  # after another test in a warm kernel the old bug did not show
def test_a_list_cell_edit_in_a_frame_held_by_a_dataclass_does_not_survive_a_rerun(nb_runner):
    nb_runner.create_notebook(
        [
            # The default threshold, which the harness pins to 0: the cell
            # that appends is then too quick to cache and runs plain, as a
            # user's would, on the very list the cache holds.
            "import time, dataclasses\nimport pandas as pd\nimport cash\n"
            "cash.configure(min_execution_time_to_cache_seconds=0.01)\n"
            "@dataclasses.dataclass\nclass Report:\n    table: object\n    n: int",
            # Slow enough that the cost model keeps the value in RAM, where
            # the frame used to be shared (0.2 s was not, measured).
            "time.sleep(0.5)\nrep = Report(pd.DataFrame({'tags': [['a'], ['b']]}), 2)",
            "rep.table['tags'].iloc[0].append('edited')",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_cell(2)
    assert nb_runner.peek("rep.table['tags'].iloc[0]").strip() == "['a']"
