"""The calls the many-cheap-calls guard samples at its re-check are still cached.

Past 50 calls the guard checks again whether a site is worth caching, on the
next few calls. Those used to run plain -- not looked up, not stored, whatever
they cost -- so ``[work(i) for i in range(n)]`` with ``work(50)`` to
``work(54)`` slow recomputed those five on every re-run while the other calls
were served.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def test_slow_calls_at_the_re_check_are_served_on_a_re_run(nb_runner, tmp_path):
    log = tmp_path / "runs.log"
    work = (
        "import os, time\n"
        f"RUNS_LOG = {str(log)!r}\n"
        "def work(i):\n"
        "    fd = os.open(RUNS_LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, f'{i} '.encode())\n"
        "    os.close(fd)\n"
        "    time.sleep(0.2 if 50 <= i < 55 else 0.01)\n"
        "    return i * 2"
    )
    nb_runner.create_notebook([work, "n = 60", "r = [work(i) for i in range(n)]"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert len(log.read_text(encoding="utf-8").split()) == 60

    nb_runner.set_cell_source(2, "n = 61")
    nb_runner.run_cells([2, 3])
    assert nb_runner.peek("sum(r)") == str(sum(i * 2 for i in range(61)))
    assert log.read_text(encoding="utf-8").split()[60:] == ["60"], "calls the cache held ran again"
