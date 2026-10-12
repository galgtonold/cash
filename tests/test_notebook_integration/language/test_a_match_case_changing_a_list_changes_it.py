"""A ``match`` whose case changes a list in place is a change to that list.

Cash runs a ``match`` as one statement, and only looked at the calls at its
top level: ``case 'train': rows.append('aug')`` was not seen, so after an
edit to the subject and Run All the cached ``slow(rows)`` below was served
for the old list.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

MATCH = (
    "mode = '{m}'\nmatch mode:\n    case 'train':\n        rows.append('aug')\n    case _:\n        rows.append('raw')"
)
CELLS = [
    "import cash\n%cash_on\nimport time\ndef slow(x):\n    time.sleep(0.3)\n    return repr(x)",
    "rows = [1, 2]",
    MATCH.format(m="train"),
    "r = slow(rows)\nprint(r)",
]


def test_run_all_after_an_edited_subject_reads_the_new_list(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[1, 2, 'aug']" in nb_runner.get_output(4), "control: the first run"
    nb_runner.set_cell_source(3, MATCH.format(m="test"))

    nb_runner.run_all()

    assert "[1, 2, 'raw']" in nb_runner.get_output(4), nb_runner.get_output(4)
