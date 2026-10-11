"""A cell that rebuilds one name in steps shows its printed output on every run.

``y = load(); y = clean(y)``: on the next Run All, and after a restart, cash
restores the last version (or finds it current) and skips the steps. Neither
step's output was shown, while a one-statement hit (``x = load()``) replays
its print.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

HELPERS = """\
import time
def load():
    time.sleep(0.2)
    print("loaded 100 rows")
    return list(range(100))
def clean(rows):
    time.sleep(0.2)
    out = [r for r in rows if r > 1]
    print(f"cleaned to {len(out)}")
    return out
"""

CELLS = ["import cash\n%cash_on", "import helpers", "x = helpers.load()", "y = helpers.load()\ny = helpers.clean(y)"]


def test_the_steps_output_is_shown_again(nb_runner):
    (nb_runner.work_dir / "helpers.py").write_text(HELPERS, encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.get_output(4) == "loaded 100 rows\ncleaned to 98"

    nb_runner.run_all()
    assert nb_runner.get_output(3) == "loaded 100 rows"
    assert nb_runner.get_output(4) == "loaded 100 rows\ncleaned to 98"

    nb_runner.settle_writes()
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.get_output(4) == "loaded 100 rows\ncleaned to 98"
    assert nb_runner.peek("len(y)") == "98"
