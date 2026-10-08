"""``importlib.reload(mylib)`` runs on every Run All, so the reader below sees the file's values.

``mylib.K = 5`` / ``importlib.reload(mylib)`` / ``x = mylib.from_k(2)``, with a
module that takes a moment to import: the second Run All served the reload
from the cache, so ``K`` stayed 5 and ``x`` was 10 where a plain kernel
gives 2.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

LIB = "import time\ntime.sleep(0.2)\nK = 1\ndef from_k(x):\n    time.sleep(0.3)\n    return K * x\n"


def test_the_second_run_all_reloads_again(nb_runner):
    (nb_runner.work_dir / "mylib.py").write_text(LIB, encoding="utf-8")
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "import mylib, importlib",
            "mylib.K = 5",
            "importlib.reload(mylib)",
            "x = mylib.from_k(2)\nprint('X', x)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()

    assert (nb_runner.peek("x"), nb_runner.peek("mylib.K")) == ("2", "1"), nb_runner.get_raw_output(4)

    nb_runner.restart()
    nb_runner.run_all()

    assert (nb_runner.peek("x"), nb_runner.peek("mylib.K")) == ("2", "1"), nb_runner.get_raw_output(4)
