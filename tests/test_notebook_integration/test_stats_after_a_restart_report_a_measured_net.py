"""A Restart & Run All reports what it saved, not a range around zero.

"%cash_stats after a restart: 'Net time saved: at least
-10.3s, at best 1.3min' ... My pair measured 80.2s saved, so the upper bound
was the right one; the lower bound tells me nothing."

The cost of each computation is now kept beside the cache, so the kernel that
restores it can still point at a measurement of what it cost.
"""

import json

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print\nimport time",
    "def slow(n):\n    time.sleep(1.2)\n    return n * 2\nv = slow(21)",
    "print('V', v)",
    "%cash_stats json",
]


def _stats(runner, cell):
    raw = runner.get_output(cell)
    return json.loads(raw[raw.index("{") : raw.rindex("}") + 1])


def test_the_net_after_a_restart_is_a_number(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "V 42" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.restart()
    nb_runner.run_all()
    assert "V 42" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    data = _stats(nb_runner, 4)
    assert data["total_measured_saved"] >= 1.0, (
        "the restart credited nothing to the measurement the first run took:\n" + nb_runner.get_output(4)
    )
    assert data["net_time_saved"] > 0, nb_runner.get_output(4)

    nb_runner.set_cell_source(4, "%cash_stats")
    nb_runner.run_cell(4)
    out = nb_runner.get_output(4)
    assert "at least" not in out, out
    assert "(measured)" in out, out
