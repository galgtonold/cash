"""The week's first day and the time zone a cell sets hold after a restart.

``calendar.setfirstweekday(6)`` and ``time.tzset()`` change a setting of
the process, which a kernel restart puts back. Restart and Run All makes
it again. So does a restart that runs only a reader below: the reader's
plan runs the setting's statement first, or ``calendar.monthcalendar``
would start its weeks on Monday and ``time.localtime`` answer in the zone
the kernel started in, where a run from the top answers otherwise.
"""

import sys

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

# A new kernel each: the setting would otherwise stay with the kernel the
# next test of this worker is given.
pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

#: name -> (the cell's statements, what a reader prints once it is made).
CASES = {
    "week-start": (
        "calendar.setfirstweekday(6)",
        "calendar.monthcalendar(2026, 11)[0]",
        "[1, 2, 3, 4, 5, 6, 7]",
    ),
    "time-zone": (
        "os.environ['TZ'] = 'JST-9'\ntime.tzset()",
        "time.localtime(0).tm_hour",
        "9",
    ),
}


def _skip_without_tzset(kind):
    if kind == "time-zone" and sys.platform == "win32":
        pytest.skip("no time.tzset on Windows")


@pytest.mark.parametrize("kind", list(CASES))
def test_the_setting_is_made_for_a_reader_run_alone_after_a_restart(nb_runner, kind):
    _skip_without_tzset(kind)
    statement, probe, made = CASES[kind]
    cells = [
        "import cash\n%cash_on",
        "import calendar, os, time",
        statement + "\ndata = list(range(10))",
        f"total = sum(data)\nprint('MADE', {probe}, total)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
        assert f"MADE {made} 45" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_cell(1)
    nb_runner.run_cell(4)

    assert f"MADE {made} 45" in nb_runner.get_output(4), nb_runner.get_raw_output(4)


def test_the_week_start_is_made_on_a_run_all_after_a_restart(nb_runner):
    """Made in a function slow enough for its statement to be worth storing.
    (The time zone's statement also writes ``TZ``, which alone keeps it
    from being stored.)"""
    kind = "week-start"
    statement, probe, made = CASES[kind]
    body = "\n".join("    " + line for line in statement.splitlines())
    cells = [
        "import cash\n%cash_on",
        f"import calendar, os, time\ndef set_up():\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n{body}\n    return 'set'",
        "done = set_up()",
        f"total = sum(range(10))\nprint('MADE', {probe}, total, done)",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
        assert f"MADE {made} 45 set" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.restart()
    nb_runner._inject_notebook_path()
    nb_runner.run_all()

    assert f"MADE {made} 45 set" in nb_runner.get_output(4), nb_runner.get_raw_output(4)
