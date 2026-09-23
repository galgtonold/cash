"""A reader that re-runs gets a fresh file handle / iterator, not an exhausted one.

Cell A opens a handle (``fh = open(p)``), cell B consumes it. On a second Run
All, plain Jupyter re-runs A and B reads the file again. Cash skipped A --
nothing about it changed -- so B, when it re-ran, iterated a handle already at
EOF and printed ``[]``. That stayed hidden while B itself was restored from
the cache; a change that stopped storing such a cheap reader
exposed it (test_file_handle_iteration_second_run_all). Here B is forced to
re-run, so caching cannot hide it.
"""

import pytest
from conftest import CASH_TEST_PIN_THRESHOLDS

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def _nb(nb_runner, producer, setup=""):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on\n%cash_badge print\n" + setup,
            "with open('probe_lines.txt', 'w') as f:\n    f.write('a\\nb\\nc\\n')\n" + producer,
            "# @cash:no-cache\nlines = [l.strip() for l in fh]\nprint('R', lines)",
        ]
    )
    nb_runner.start_kernel()


@pytest.mark.parametrize(
    "producer",
    [
        "fh = open('probe_lines.txt')",
        "fh = iter(open('probe_lines.txt').read().splitlines())",
    ],
)
def test_a_second_run_all_reads_the_data_again(nb_runner, producer):
    _nb(nb_runner, producer)
    nb_runner.run_all()
    assert "R ['a', 'b', 'c']" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
    nb_runner.run_all()
    assert "R ['a', 'b', 'c']" in nb_runner.get_output(3), (
        "the reader re-ran against the iterator the first run exhausted:\n"
        + nb_runner.get_raw_output(2)
        + nb_runner.get_raw_output(3)
    )


def test_re_running_only_the_reader_reads_the_data_again(nb_runner):
    """Plain Jupyter would print [] here (the handle is at EOF and cell A did
    not run); Restart & Run All prints the lines. cash promises the latter."""
    _nb(nb_runner, "fh = open('probe_lines.txt')")
    nb_runner.run_all()
    nb_runner.run_cell(3)
    assert "R ['a', 'b', 'c']" in nb_runner.get_output(3), nb_runner.get_raw_output(3)


def test_a_slow_open_is_not_served_to_the_next_run(nb_runner):
    """With the cost floors at zero every call is worth storing, as ``open``
    was on a loaded machine: the first run stored the handle as an
    intercepted call and the second Run All served it, drained, to the reader.
    """
    _nb(nb_runner, "fh = open('probe_lines.txt')", setup=CASH_TEST_PIN_THRESHOLDS)
    nb_runner.run_all()
    assert "R ['a', 'b', 'c']" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
    nb_runner.run_all()
    assert "R ['a', 'b', 'c']" in nb_runner.get_output(3), (
        "a stored file handle was served to the reader:\n" + nb_runner.get_raw_output(2) + nb_runner.get_raw_output(3)
    )
