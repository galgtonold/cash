"""A slow call handed the object its statement changes is cached on its value.

``df.insert(1, "c", slow(df))`` changes ``df`` in place, so the statement
runs every time. ``slow(df)`` inside it is the work, and it is keyed on the
frame it receives: on an unchanged run it is served from the cache and the
statement puts the stored column in. The same holds in a loop body
(``for d in dfs: d["c"] = slow(d)``).

Oracle: an external counter file, written by ``slow`` on every real run.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\n%cash_badge print\nimport time\nimport pandas as pd"


def _slow(counter):
    return (
        "def slow(d, k=2):\n"
        f"    open(r'{counter}', 'a').write('X')  # @cash:assume-safe (a test counter)\n"
        "    time.sleep(0.3)\n"
        "    return d['a'] * k"
    )


FRAME = "df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})"

CASES = {
    "insert": ([FRAME, "df.insert(1, 'c', slow(df))"], "print('C', list(df['c']))", "C [2, 4, 6, 8, 10]", 1),
    "loop_body": (
        ["dfs = [pd.DataFrame({'a': [1, 2]}), pd.DataFrame({'a': [3, 4]})]", "for d in dfs:\n    d['c'] = slow(d)"],
        "print('C', [list(d['c']) for d in dfs])",
        "C [[2, 4], [6, 8]]",
        2,
    ),
}


def _runs(counter):
    return len(counter.read_bytes()) if counter.exists() else 0


@pytest.mark.parametrize("case", list(CASES))
def test_an_unchanged_run_all_reuses_the_slow_call(nb_runner, tmp_path, case):
    cells, reader, expect, cold_runs = CASES[case]
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook([SETUP, _slow(counter), *cells, reader])
    nb_runner.start_kernel()
    last = len(cells) + 3
    nb_runner.run_all()
    assert expect in nb_runner.get_output(last)
    assert _runs(counter) == cold_runs

    for _ in range(2):
        nb_runner.run_all()
        assert expect in nb_runner.get_output(last), nb_runner.get_output(last)
    nb_runner.restart()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(last), nb_runner.get_output(last)
    assert _runs(counter) == cold_runs, nb_runner.get_output(last - 1)


def test_a_changed_frame_runs_the_call_again(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    nb_runner.create_notebook(
        [SETUP, _slow(counter), FRAME, "df.insert(1, 'c', slow(df))", "print('C', list(df['c']))"]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "df = pd.DataFrame({'a': [1, 2, 3, 4, 50]})")
    nb_runner.run_all()
    assert "C [2, 4, 6, 8, 100]" in nb_runner.get_output(5), nb_runner.get_output(5)
    assert _runs(counter) == 2


def test_a_call_that_changes_the_frame_it_is_handed_still_changes_it(nb_runner, tmp_path):
    counter = tmp_path / "calls.log"
    helper = (
        "def tag(d):\n"
        f"    open(r'{counter}', 'a').write('X')  # @cash:assume-safe (a test counter)\n"
        "    time.sleep(0.3)\n"
        "    d['seen'] = 1\n"
        "    return d['a'] * 2"
    )
    nb_runner.create_notebook([SETUP, helper, FRAME, "df.insert(1, 'c', tag(df))", "print('COLS', list(df.columns))"])
    nb_runner.start_kernel()
    for _ in range(3):
        nb_runner.run_all()
        assert "COLS ['a', 'c', 'seen']" in nb_runner.get_output(5), nb_runner.get_output(5)
