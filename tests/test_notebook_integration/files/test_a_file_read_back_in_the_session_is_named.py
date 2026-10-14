"""A report written, opened again, saved and read back to check it is named
once (``NOTEBOOK-FILE-ROUNDTRIP``).

The shape of a formatting pass over an Excel report: one cell writes the
file, the next reads it, writes it again and reads it back. Every write gives
the file new bytes, so the read back runs again on every Run All. The second
Run All says so, naming the file and both statements; the third says nothing
more.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CODE = "[NOTEBOOK-FILE-ROUNDTRIP]"

CELLS = [
    "%cash_badge print\nimport pathlib, time\nREPORT = 'report.txt'",
    "pathlib.Path(REPORT).write_text(str(time.time_ns()))",
    "text = pathlib.Path(REPORT).read_text()\n"
    "pathlib.Path(REPORT).write_text(text + ' formatted')\n"
    "check = [pathlib.Path(REPORT).read_text(), time.sleep(1.1)][0]\n"
    "print(check.endswith('formatted'))",
]


def test_a_report_read_back_on_every_run_is_named_once(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_raw_output(3)
    assert "True" in first and CODE not in first, first

    nb_runner.run_all()
    second = nb_runner.get_raw_output(3)
    assert CODE in second, second
    assert "report.txt" in second
    assert "written by `pathlib.Path(REPORT).write_text(text + ' formatted')`" in second, second
    assert "read back by `check = [pathlib.Path(REPORT).read_text(), time.sleep(1.1)][0]`" in second, second

    nb_runner.run_all()
    third = nb_runner.get_raw_output(3)
    assert "True" in third and CODE not in third, third
