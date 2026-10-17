"""What is built on a file rewritten with the same bytes restores after a
restart.

A data export ran again and rewrote ``meters.csv`` unchanged. In the same
kernel the reader came back from the cache; after the restart it ran again
(too quick to keep on disk), under the file's new stat, and every statement
built on it missed: a t-SNE ran again for 8 s with its input identical. The
lineage names a file rewritten unchanged by its earlier version.
"""

import os

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

CELLS = [
    "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
    "import time",
    "rows = open('meters.csv').read().splitlines()",
    "n = (time.sleep(0.6), len(rows))[1]\nprint('N', n)",
]


def test_the_reader_of_a_rewritten_file_restores_after_a_restart(nb_runner):
    data = nb_runner.work_dir / "meters.csv"
    data.write_text("a,b\n1,2\n3,4\n", encoding="utf-8", newline="\n")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "N 3" in nb_runner.get_output(4)

    before = os.stat(data).st_mtime_ns
    data.write_text("a,b\n1,2\n3,4\n", encoding="utf-8", newline="\n")
    os.utime(data, ns=(before + 2_000_000_000, before + 2_000_000_000))
    nb_runner.run_cells([3, 4])
    assert shows_cached(nb_runner.get_output(4)), f"control: the same bytes\n{nb_runner.get_output(4)}"

    nb_runner.restart()
    nb_runner.run_all()

    out = nb_runner.get_output(4)
    assert "N 3" in out, out
    assert shows_cached(out), out


WRITE_THEN_READ = [
    "import cash\n%load_ext cash\n%cash_badge print\n%cash_on",
    "import time",
    # ``tarfile.extractall`` then ``pd.read_csv``: the statement writes a
    # file, so it is never stored, and it rewrites the file on every run.
    "def load():\n"
    "    with open('meters.csv', 'w', newline='\\n') as fh:\n"
    "        fh.write('a,b\\n1,2\\n3,4\\n')\n"
    "    return open('meters.csv').read().splitlines()\n"
    "rows = load()",
    "n = (time.sleep(0.6), len(rows))[1]\nprint('N', n)",
]


def test_a_file_the_loading_statement_rewrites_keeps_its_readers_after_a_restart(nb_runner):
    nb_runner.create_notebook(WRITE_THEN_READ)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "N 3" in nb_runner.get_output(4)
    nb_runner.settle_writes()

    nb_runner.restart()
    nb_runner.run_all()

    assert "NOT CACHED" in nb_runner.get_output(3), nb_runner.get_output(3)
    out = nb_runner.get_output(4)
    assert "N 3" in out, out
    assert shows_cached(out), out
