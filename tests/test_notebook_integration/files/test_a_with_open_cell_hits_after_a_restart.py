"""``with open(p) as fh: text = fh.read()`` is stored to disk and hits after a restart, quietly.

The closed file ``fh`` is one of the statement's outputs and a file object
does not pickle: the disk write failed, ``[TIERED] Failed to write to
backend FileBackend: cannot pickle 'TextIOWrapper' instances`` was printed
in the cell's output on every run, and the cell ran again after a restart.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]


def test_hits_after_a_restart(nb_runner, tmp_path):
    (tmp_path / "data.txt").write_text("hello world\n", encoding="utf-8")
    cells = [
        "%cash_badge print\nimport time",
        "with open('data.txt') as fh:\n    text = fh.read()\n    time.sleep(0.3)\nprint(len(text))",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(2)
    assert "12" in raw and "Failed to write" not in raw, raw

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    raw = nb_runner.get_raw_output(2)
    assert "CACHED" in raw and "Failed to write" not in raw, raw
    assert nb_runner.peek("(type(fh).__name__, fh.closed, fh.name)") == "('TextIOWrapper', True, 'data.txt')"
