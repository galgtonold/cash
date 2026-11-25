"""After a kernel restart, a cell run alone reads a notebook variable as the functions called above left it.

``results = []``, ``def record(x): results.append(...)``, ``n = record(21)``,
then ``print(results)``: Run All, restart, run only the last cell. The cell
printed ``[]``: the list was rebuilt from its first assignment and none of
the calls that filled it ran again, silently, where Restart & Run All prints
``[42]``. The same for a dict or a counter a helper updates, a registry a
call fills, a helper called in a loop and a ``key=`` function that logs.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300), pytest.mark.fresh_kernel]

SETUP = "import time"

CASES = {
    "a_list_a_slow_helper_appends_to": (
        [
            "results = []",
            "def record(x):\n    time.sleep(0.3)\n    results.append(x * 2)\n    return len(results)",
            "n = record(21)",
        ],
        "print('R', results)",
        "R [42]",
    ),
    "a_list_two_cells_append_to": (
        ["LOG = []\ndef rec(v):\n    LOG.append(v)\n    time.sleep(0.3)\n    return v * 2", "a = rec(1)", "b = rec(2)"],
        "c = len(LOG)\nprint('R', c, LOG)",
        "R 2 [1, 2]",
    ),
    "a_dict_counter_a_helper_updates": (
        [
            "stats = {'n': 0}\ndef work(x):\n    stats['n'] += 1\n    time.sleep(0.3)\n    return x * 2",
            "l1 = work(1)",
            "l2 = work(2)",
        ],
        "done = stats['n']\nprint('R', done, stats)",
        "R 2 {'n': 2}",
    ),
    "a_counter_rebound_under_global": (
        [
            "step = 0\ndef train_one(x):\n    global step\n    step += 1\n    time.sleep(0.3)\n    return x * 2",
            "l1 = train_one(1)",
            "l2 = train_one(2)",
        ],
        "seen = step\nprint('R', seen)",
        "R 2",
    ),
    "a_registry_a_bare_call_fills": (
        [
            "registry = []\ndef register(f):\n    registry.append(f.__name__)\n    return f",
            "def a(): pass\nregister(a)",
        ],
        "print('R', registry)",
        "R ['a']",
    ),
    "a_helper_called_in_a_loop": (
        ["LOG = []\ndef note(v):\n    LOG.append(v)", "for i in range(3):\n    note(i)"],
        "print('R', LOG)",
        "R [0, 1, 2]",
    ),
    "a_bare_call_bumping_a_global": (
        ["cnt = 0\ndef bump():\n    global cnt\n    cnt += 1", "bump()"],
        "print('R', cnt)",
        "R 1",
    ),
    "a_key_function_that_logs": (
        ["seen = []\ndef k(v):\n    seen.append(v)\n    return v", "s = sorted([3, 1, 2], key=k)"],
        "print('R', seen)",
        "R [3, 1, 2]",
    ),
    "a_reader_that_calls_the_helper_again": (
        ["LOG = []\ndef rec(v):\n    LOG.append(v)\n    time.sleep(0.3)\n    return v * 2", "a = rec(1)", "b = rec(2)"],
        "c = rec(3)\nprint('R', c, LOG)",
        "R 6 [1, 2, 3]",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_reader_run_alone_after_a_restart_sees_what_the_calls_above_changed(nb_runner, case):
    above, reader, expected = CASES[case]
    cells = [SETUP, *above, reader]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expected in nb_runner.get_output(len(cells)), "control: the top-to-bottom run"

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_cell(len(cells))

    assert expected in nb_runner.get_output(len(cells)), nb_runner.get_raw_output(len(cells))


def test_a_cell_written_after_the_restart_continues_from_the_calls_above(nb_runner):
    """No run of the new cell's statement left anything to read back: what
    it changes is read from the helper's source alone."""
    cells = [SETUP, "LOG = []\ndef rec(v):\n    LOG.append(v)\n    return v * 2", "a = rec(1)", "b = rec(2)", "x = 1"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.set_cell_source(5, "c = rec(3)\nprint('R', c, LOG)")
    nb_runner.run_cell(5)

    assert "R 6 [1, 2, 3]" in nb_runner.get_output(5), nb_runner.get_raw_output(5)
