"""A cell calling a function defined in a cell that reads an environment variable follows the variable.

``def mode(): ... return os.environ.get("MODE")``, a cell setting ``MODE``
and ``x = mode() * 2``: editing the setting and running it and the caller,
or Restart & Run All, served the first mode with no warning. The
environment a called function reads was keyed only for functions of the
user's modules.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

DEF = "import os, time\ndef mode():\n    time.sleep(0.3)\n    return os.environ.get('{var}', '-')"


@pytest.fixture
def var(tmp_path):
    """A name of its own: the warm kernel keeps its environment between tests."""
    return f"CASH_IT_NB_MODE_{abs(hash(str(tmp_path))) % 10**8}"


def test_editing_the_setting(nb_runner, var):
    nb_runner.create_notebook([DEF.format(var=var), f"os.environ['{var}'] = 'a'", "x = mode() * 2\nprint('X', x)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "X aa" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.set_cell_source(2, f"os.environ['{var}'] = 'b'")
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert "X bb" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.set_cell_source(2, f"os.environ['{var}'] = 'c'")
    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()
    assert "X cc" in nb_runner.get_output(3), nb_runner.get_raw_output(3)
