"""A cell's function handed to a ``@cash.cache`` function of the user's module
runs on every Run All, as plain Python runs it.

``y = mylib.run(record, 3)`` with ``record`` appending to ``LOG``: the cell
re-ran on the second Run All, but the cached call was a hit, so ``record``
never ran and ``LOG`` stayed empty.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(180)]

MYLIB = "import cash\n\n@cash.cache\ndef run(f, v):\n    return f(v)\n"


def test_the_handed_function_runs_on_every_run_all(nb_runner, tmp_path):
    (tmp_path / "mylib.py").write_text(MYLIB, encoding="utf-8")
    cells = [
        f"import os, sys\nos.chdir(r'{tmp_path}')\nsys.path.insert(0, r'{tmp_path}')\n"
        "import mylib\nLOG = []\ndef record(v):\n    LOG.append(v)\n    return v * 2",
        "y = mylib.run(record, 3)",
        "print('ROWS', len(LOG), y)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
        assert "ROWS 1 6" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
