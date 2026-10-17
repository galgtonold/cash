"""``from m import *`` runs again after a restart instead of being served.

A slow star import (sympy takes about 0.25 s) was stored, and after a restart
the hit restored it. The names a star import binds are known only by running
it, so the restore bound none of them and the next cell raised NameError
where a plain kernel prints the result.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.modules, pytest.mark.timeout(300)]

LIB = "import time\ntime.sleep(0.5)\n\ndef answer():\n    return 42\n"


def test_names_from_a_slow_star_import_exist_after_a_restart(nb_runner, tmp_path):
    (tmp_path / "slowlib.py").write_text(LIB, encoding="utf-8")
    nb_runner.create_notebook(["from slowlib import *", "print('A', answer())"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "A 42" in nb_runner.get_output(2)

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()

    assert "A 42" in nb_runner.get_output(2), nb_runner.get_raw_output(2)
