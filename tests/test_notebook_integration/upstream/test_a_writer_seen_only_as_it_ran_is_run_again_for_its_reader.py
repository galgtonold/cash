"""A writer cash only SAW writing -- a helper's ``gzip.open(p, 'wt')``, which
no writer name shows -- is run again for the cell reading its file, after an
edit above and a run of that cell alone, as Run All would.

The writer scheduler found writers and their paths from the source only: the
reader got the old file's content, and an export cell replayed above it,
which clears the folder, deleted the file for good. A reader whose call does
not show the path either (``gzip.open(p, 'rt')``, ``np.loadtxt(p)``) counts by
what it read when it last ran.
"""

import pytest

pytestmark = [pytest.mark.integration]

SETUP = """import os, glob, gzip, time
import numpy as np
X = [1.0, 2.0]
OUT = 'export'
os.makedirs(OUT, exist_ok=True)
def export_total(a, path):
    time.sleep(0.2)
    f = gzip.open(path, 'wt'); f.write(str(float(sum(a)))); f.close()
    return len(a)
"""


@pytest.mark.parametrize(
    ("writer", "reader", "printed"),
    [
        ("n = export_total(X, 'stage.gz')", "tot = gzip.open('stage.gz', 'rt').read()", "30.0"),
        ("np.savetxt('arr.txt', np.array(X))", "tot = np.loadtxt('arr.txt').sum()", "30.0"),
    ],
    ids=["helper-gzip", "loadtxt-reader"],
)
def test_the_reader_gets_the_new_content(nb_runner, writer, reader, printed):
    nb_runner.create_notebook([SETUP, writer, reader + "\nprint('read', tot)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(1, SETUP.replace("X = [1.0, 2.0]", "X = [10.0, 20.0]"))
    nb_runner.run_cell(3)
    assert nb_runner.get_output(3).strip().endswith(f"read {printed}")


def test_a_replayed_clearing_does_not_lose_the_file(nb_runner):
    cells = [
        SETUP.replace("X = [1.0, 2.0]", "REGIONS = ['north', 'south']\nX = [1.0, 2.0]"),
        "for old in glob.glob(os.path.join(OUT, '*')):\n    os.remove(old)\n"
        "for reg in REGIONS:\n    open(os.path.join(OUT, f'{reg}.csv'), 'w').write(reg)",
        "n = export_total(X, os.path.join(OUT, 'summary.gz'))",
        "print(sorted(os.listdir(OUT)))",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(1, cells[0].replace("['north', 'south']", "['north', 'east']"))
    nb_runner.run_cell(4)
    assert nb_runner.get_output(4).strip().endswith("['east.csv', 'north.csv', 'summary.gz']")
