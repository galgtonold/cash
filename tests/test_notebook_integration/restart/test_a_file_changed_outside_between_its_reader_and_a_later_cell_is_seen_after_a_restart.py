"""A file changed outside the notebook between the cell that read it and a
later cell built on it makes the later cell stale after a restart.

The first cell reads the file and writes another, so it is never stored;
the later cell is, with the read file among its dependencies through what
it was built on. The file is then rewritten from outside, with the same
size and its old modification time put back. After a restart, Run All must
give what plain Python gives: the later cell runs again on the new bytes.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.restore]


def _cells(data, log):
    return [
        "import cash\n%cash_on\n%cash_badge print",
        f"text = [open({str(data)!r}).read(), open({str(log)!r}, 'w').write('x')][0]",
        "import time\nshout = (time.sleep(0.3), text.upper())[1]",
        "print('REPORT', shout)",
    ]


def test_a_file_changed_outside_makes_the_later_cell_stale(nb_runner, tmp_path):
    data, log = tmp_path / "data.txt", tmp_path / "log.txt"
    data.write_text("abc", encoding="utf-8")
    nb_runner.create_notebook(_cells(data, log))
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "REPORT ABC" in nb_runner.get_output(4)

    nb_runner.restart()
    nb_runner.run_all()
    assert "REPORT ABC" in nb_runner.get_output(4)
    out = nb_runner.get_output(3)
    assert "CACHED: shout" in out, "control: the later cell is stored: " + out

    before = os.stat(data)
    data.write_text("xyz", encoding="utf-8")  # outside the notebook, same size
    os.utime(data, ns=(before.st_atime_ns, before.st_mtime_ns))
    nb_runner.restart()
    nb_runner.run_all()
    assert "REPORT XYZ" in nb_runner.get_output(4), nb_runner.get_output(3)
    assert "CACHED: shout" not in nb_runner.get_output(3)
