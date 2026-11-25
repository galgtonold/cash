"""After a kernel restart, a cell run alone runs with the settings the cells above wrote as an assignment or a ``del``.

``getcontext().prec = 6`` (how Decimal's precision is set),
``logging.getLogger('noisy').disabled = True`` and ``del os.environ['K']``
bind no name and are no call. The rebuild runs again the statements above
that changed the process, and looked for them among the statements the
simulation keeps, which left these out: the cell computed with the default
precision, the logger enabled and the variable set, silently, where Restart
& Run All gives the notebook's settings.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

CASES = {
    "decimal_precision": (
        [
            "import time\nfrom decimal import Decimal, getcontext\ngetcontext().prec = 6",
            "time.sleep(0.3)\nprice = Decimal('19.99')",
        ],
        "r = str(price / 7)\nprint('R', r)",
        "R 2.85571",
    ),
    "a_logger_disabled": (
        ["import time, logging\nlogging.getLogger('noisy').disabled = True", "time.sleep(0.3)\nx = 1"],
        "r = logging.getLogger('noisy').isEnabledFor(logging.ERROR)\nprint('R', r)",
        "R False",
    ),
    "an_environment_variable_deleted": (
        [
            "import time, os\nos.environ['CASH_TEST_PROXY'] = 'p'",
            "time.sleep(0.3)\ndel os.environ['CASH_TEST_PROXY']\nx = 1",
        ],
        "import os as _o\nr = _o.getenv('CASH_TEST_PROXY')\nprint('R', r)",
        "R None",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_reader_run_alone_after_a_restart_runs_with_the_setting(nb_runner, case):
    above, reader, expected = CASES[case]
    cells = [*above, reader]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expected in nb_runner.get_output(len(cells)), "control: the top-to-bottom run"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(len(cells))

    assert expected in nb_runner.get_output(len(cells)), nb_runner.get_raw_output(len(cells))
