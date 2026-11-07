"""A cell re-run above a cell that binds its input anew says so, also when it fails.

Found by replaying a real session (JuNE student_2): the cell splitting the
log's lines (``lines2 = []``, a loop appending the parts, ``lines = lines2``)
sat below the cell parsing them. Re-run on its own after the split, the parse
cell got ``lines`` as the cells above leave it, as a run from the top does:
unsplit, and the parse raised a ValueError plain Jupyter (on the split lines)
never raised. The warning named an in-place change through ``lines2``, or
with a plain ``lines = [...]`` below, nothing at all. It now names the
binding below, next to the error.
"""

import pytest
from nbclient.exceptions import CellExecutionError

pytestmark = [pytest.mark.timeout(120), pytest.mark.upstream]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
WRITE_LOG = (
    "with open('log.txt', 'w') as fh:\n"
    "    for i in range(50):\n"
    "        fh.write('u%d:%d@u%d:%d\\n' % (i % 7, i, i % 5, 3 * i))"
)
READ = "f = open('log.txt')\nlines = f.readlines()\nlen(lines)"
PARSE = "parsed = [int(l.split(':')[1]) for l in lines]\nprint('PARSED', len(parsed))"
SPLIT = (
    "lines2 = []\n"
    "for line in lines:\n"
    "    for l in line.strip().replace('@', '\\n@').split('\\n'):\n"
    "        lines2.append(l)\n"
    "lines = lines2\n"
    "len(lines)"
)
COMPREHENSION = "lines = [x for l in lines for x in l.strip().replace('@', '\\n@').split('\\n')]\nlen(lines)"


@pytest.mark.parametrize("split", [SPLIT, COMPREHENSION], ids=["through_lines2", "bound_directly"])
def test_the_parse_cell_rerun_above_the_split_names_it_with_its_error(nb_runner, split):
    nb_runner.create_notebook([SETUP, WRITE_LOG, READ, PARSE, split])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3, 5])

    with pytest.raises(CellExecutionError) as raised:
        nb_runner.run_cell(4)  # a run from the top raises here too
    text = str(raised.value)
    assert raised.value.ename == "ValueError", text
    assert "NOTEBOOK-CHANGED-BELOW" in text, text
    assert "cell 5 below binds anew (`lines = " in text, text


def test_the_split_above_the_parse_cell_is_quiet(nb_runner):
    """Positive control: the split above the parse cell, as the file now has it."""
    nb_runner.create_notebook([SETUP, WRITE_LOG, READ, SPLIT, PARSE])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3, 4, 5, 5])
    output = nb_runner.get_output(5)
    assert "PARSED 100" in output, output
    assert "NOTEBOOK-CHANGED-BELOW" not in output, output
