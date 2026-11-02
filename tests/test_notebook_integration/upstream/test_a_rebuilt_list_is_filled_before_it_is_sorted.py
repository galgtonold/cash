"""A list emptied again to bring a cell up to date is filled again too.

Found by replaying a real session with the notebook known to cash. Its cells
read a log out of a zip, parse it into ``data`` (``data = []`` then a loop
appending to it), sort ``data`` in place, and build a frame from it. After a
cell was inserted above the frame cell and that cell run again, the next cell
reading the frame got an empty one: to rebuild the frame cash re-ran
``data = []`` and the sort, but not the loop between them, so every cell
below counted no rows. A run from the top has the full list there.

The sort changes ``data`` in place, and the rule that re-runs a loop's
accumulator init with the loop took the sort for that loop. A re-run init
now brings along every statement between it and the next statement re-run
that changes the list.
"""

import pytest

pytestmark = [pytest.mark.timeout(120), pytest.mark.upstream]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
WRITE_LOG = (
    "import random, zipfile\n"
    "random.seed(1)\n"
    "lines = ['@User%d: ' % (i % 50) + ' -> '.join("
    "'Action_%d (%02d/0%d/20 | %02d:%02d:00)' % (random.randrange(5), 1 + i % 27, 1 + i % 9, h, random.randrange(60))"
    " for h in range(1 + i % 4)) for i in range(5000)]\n"
    "with zipfile.ZipFile('logs.zip', 'w') as z:\n"
    "    z.writestr('logs.txt', '\\n'.join(lines + ['']))"
)
READ_LOG = (
    "import io, zipfile\n"
    "with zipfile.ZipFile('logs.zip') as z:\n"
    "    with io.TextIOWrapper(z.open('logs.txt'), encoding='utf-8') as f:\n"
    "        data_raw = f.read().split('\\n')"
)
PARSE = (
    "from datetime import datetime\n\n"
    "data = []\n"
    "for row in data_raw:\n"
    "    if not row:\n"
    "        continue\n"
    "    i = row.find(':')\n"
    "    actions = []\n"
    "    for a in row[i + 1:].split('->'):\n"
    "        j, k = a.find('('), a.find(')')\n"
    "        actions.append((a[:j].strip(), datetime.strptime(a[j:k + 1], '(%d/%m/%y | %H:%M:%S)')))\n"
    "    data.append([int(row[5:i]), actions])"
)
SORT = "data.sort(key=lambda x: x[-1][0][-1])"
FRAME = (
    "import pandas as pd\n"
    "rows = []\n"
    "for sid, (uid, actions) in enumerate(data):\n"
    "    for name, t in actions:\n"
    "        rows.append((uid, sid, t, name))\n"
    "df = pd.DataFrame(rows, columns=['user_id', 'session_num', 'action_time', 'action_name'])"
)
READER = "print('ROWS', len(df))"


def test_a_cell_below_reads_every_row_after_a_cell_is_inserted_above_the_frame(nb_runner):
    nb_runner.create_notebook([SETUP, WRITE_LOG, READ_LOG, PARSE, SORT, FRAME, READER])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2, 3, 4, 5, 6])
    rows = nb_runner.peek("len(df)")
    assert rows not in ("0", "?"), rows

    nb_runner.insert_cell(6, "")
    nb_runner.run_cells([7, 8])

    assert f"ROWS {rows}" in nb_runner.get_output(8), nb_runner.get_output(8)
    assert nb_runner.peek("len(data)") == "5000"
