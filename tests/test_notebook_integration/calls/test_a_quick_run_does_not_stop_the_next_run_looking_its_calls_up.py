"""A quick run of a statement does not stop a later run looking its calls up.

The "not worth routing" mark of a statement that ran under the call cost
floor was kept under its text: one quick run, or one quick loop iteration,
and the next run -- with other inputs -- neither looked its calls up nor
stored them. ``cfg`` switched to a quick value and back recomputed the slow
call the cache held; a loop whose items alternate quick and slow recomputed
every slow item after a quick one on each re-run.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def _counting(name: str, body: str) -> str:
    """A function *name* that appends its first argument's tag to RUNS_LOG."""
    return (
        "import os, time\n"
        f"def {name}(item):\n"
        "    fd = os.open(RUNS_LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)\n"
        "    os.write(fd, (str(item[0]) + ' ').encode())\n"
        "    os.close(fd)\n"
        f"{body}"
    )


def _setup(nb_runner, tmp_path, cells):
    log = tmp_path / "runs.log"
    nb_runner.create_notebook([f"RUNS_LOG = {str(log)!r}", *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    return log


def test_a_parameter_switched_to_a_quick_value_and_back_is_served(nb_runner, tmp_path):
    run = _counting("run", "    time.sleep(item[1])\n    return item[1] * 10")
    log = _setup(nb_runner, tmp_path, [run, "cfg = ('a', 0.3)", "res = run(cfg) * 2"])
    for cfg in ("('b', 0.0)", "('a', 0.3)"):
        nb_runner.set_cell_source(3, f"cfg = {cfg}")
        nb_runner.run_cells([3, 4])
    assert nb_runner.peek("res") == "6.0"
    assert log.read_text(encoding="utf-8").split() == ["a", "b"], "run(cfg) for 'a' was computed again"


def test_a_loop_alternating_quick_and_slow_items_serves_the_slow_ones(nb_runner, tmp_path):
    process = _counting("process", "    time.sleep(item[1])\n    return item[0].upper()")
    items = [("a", 0), ("b", 0.2), ("c", 0), ("d", 0.2), ("e", 0), ("f", 0.2)]
    loop = "out = []\nfor it in items:\n    out.append(process(it))"
    log = _setup(nb_runner, tmp_path, [process, f"items = {items}", loop])
    assert log.read_text(encoding="utf-8").split() == ["a", "b", "c", "d", "e", "f"]
    nb_runner.set_cell_source(3, f"items = {items + [('g', 0)]}")
    nb_runner.run_cells([3, 4])
    assert nb_runner.peek("out") == repr(list("ABCDEFG"))
    slow_again = [tag for tag in log.read_text(encoding="utf-8").split()[6:] if tag in ("b", "d", "f")]
    assert slow_again == [], f"slow items computed again: {slow_again}"
