"""An ``if`` in a loop reads the frame its branch may change only when the
branch runs.

``for k in range(7): if df.ss[k] is None: df.ss[k] = df.ss_mew[k]`` gave
``df`` a new lineage from a hash of the whole frame after every pass, whether
the branch ran or not: 23 s against 9 ms plain for a 1M-row frame. A branch
that does run still moves the frame on, so a cell reading it afterwards is
not served the value from before.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
# Installed afresh on every use, over the real function: cash's modules
# outlive a test in a reused kernel, and a sibling test wraps the same one.
COUNT = (
    "import cash.notebook.control_structures.helpers as h\n"
    "h.update_mutated_variable_lineages = getattr(h, 'real_update', h.update_mutated_variable_lineages)\n"
    "h.real_update = h.update_mutated_variable_lineages\n"
    "h.updated = []\n"
    "def counting(shell, sp, names, *a, h=h, **k):\n"
    "    h.updated.extend(sorted(names))\n"
    "    return h.real_update(shell, sp, names, *a, **k)\n"
    "h.update_mutated_variable_lineages = counting\n"
)
UNCOUNT = (
    "import cash.notebook.control_structures.helpers as h\n"
    "if hasattr(h, 'real_update'):\n"
    "    h.update_mutated_variable_lineages = h.real_update\n"
    "    del h.real_update\n"
)
UPDATED = "__import__('cash.notebook.control_structures.helpers').notebook.control_structures.helpers.updated"


def test_a_branch_that_never_runs_leaves_the_frame_alone(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            "import pandas as pd\ndf = pd.DataFrame({'ss': ['a'] * 1000, 'ss_mew': ['b'] * 1000})\nmissing = 'zz'",
            "for k in range(7):\n    if df.ss[k] == missing:\n        df.loc[k, 'ss'] = df.ss_mew[k]",
            "n_b = int((df.ss == 'b').sum())\nprint('NB', n_b)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    nb_runner.peek(f"exec({COUNT!r})")
    try:
        nb_runner.run_cells([3, 4])
        assert "NB 0" in nb_runner.get_output(4)
        assert nb_runner.peek(f"{UPDATED}.count('df')") == "0"

        # The branch runs for one row: the frame moves on, and the cell after it sees the change.
        nb_runner.set_cell_source(
            2, "import pandas as pd\ndf = pd.DataFrame({'ss': ['a'] * 1000, 'ss_mew': ['b'] * 1000})\nmissing = 'a'"
        )
        nb_runner.run_cells([2, 3, 4])
        assert "NB 7" in nb_runner.get_output(4)
        assert nb_runner.peek("int((df.ss == 'b').sum())") == "7"
    finally:
        nb_runner.peek(f"exec({UNCOUNT!r})")
