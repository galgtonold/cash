"""A cell reads the binding a later cell above made, not the loop's frame.

Found replaying a real session (JuNE student_2) with the notebook known to
cash: a cell builds a frame from lists a loop fills, a cell below it binds
``df`` to a small example frame, and the user ran the loop cell again before
the cell under both. A run from the top hands that cell the example frame (a
``KeyError`` on its column there); cash kept the loop's frame in memory, as
the trust in a loop-derived value whose code is above took it for current
without asking whether a later statement above binds the name anew.
"""

import pytest

pytestmark = [pytest.mark.timeout(120), pytest.mark.upstream]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
BUILD = (
    "import pandas as pd\n"
    "a_col = []\n"
    "b_col = []\n"
    "for i in range(50):\n"
    "    a_col.append(i % 7)\n"
    "    b_col.append(i)\n"
    "df = pd.DataFrame({'user_id': a_col, 'b': b_col})"
)
EXAMPLE = "import pandas as pd\ndf = pd.DataFrame({'time': [1, 2, 3]})"
USE = "print('COLS', list(df.columns))"


def test_the_cell_below_reads_the_later_binding(nb_runner):
    nb_runner.create_notebook([SETUP, BUILD, EXAMPLE, USE])
    nb_runner.start_kernel()
    nb_runner.run_cells([2, 3, 4, 2, 4])

    assert "COLS ['time']" in nb_runner.get_output(4), nb_runner.get_output(4)


def test_a_loop_frame_nothing_rebinds_below_is_kept(nb_runner):
    nb_runner.create_notebook([SETUP, EXAMPLE, BUILD, USE])
    nb_runner.start_kernel()
    nb_runner.run_cells([2, 3, 4, 4])

    output = nb_runner.get_output(4)
    assert "COLS ['user_id', 'b']" in output, output
    assert "Upstream:" not in output, output
