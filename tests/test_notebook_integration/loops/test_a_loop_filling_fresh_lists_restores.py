"""A loop run as one unit that fills lists it seeded empty restores on a re-run.

``deltas = []`` then ``for g in groups: for d in g: deltas.append(d)``, and a
loop with ``continue`` filling four lists seeded by one tuple assignment: both
were refused as in-place mutations, so they re-ran on every run and nothing
downstream of the lists was cached either (student 2 and student 6 of the
JuNE notebooks: 36 s per run of the student 6 cell, 2.2 s with the loop
restored). The containers start empty in the same cell, so the loop's result
depends on its inputs alone and the unit can be stored like a comprehension.

A list that held something before the loop, or that another name shares,
still re-runs, and the result always matches a plain kernel.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops, pytest.mark.mutations, pytest.mark.timeout(180)]

SETUP = "import cash\n%cash_on\n%cash_badge print"
GROUPS = "groups = [(i, i + 1, i + 2) for i in range({n})]"
NESTED = (
    "deltas = []\n"
    "for g in groups:\n"
    "    w = sum(range(20000))  # work worth storing\n"
    "    for d in g:\n"
    "        deltas.append(d * 2)\n"
    "print('N', len(deltas), sum(deltas), g, d)"
)
CONTINUE = (
    "evens, odds, idx, seen = [], [], [], set()\n"
    "count = 0\n"
    "for i, g in enumerate(groups):\n"
    "    if i % 3 == 0:\n"
    "        continue\n"
    "    count += 1\n"
    "    w = sum(range(200000))  # work worth storing\n"
    "    for v in g:\n"
    "        if v % 2 == 0:\n"
    "            evens.append(v)\n"
    "        else:\n"
    "            odds.append(v)\n"
    "        seen.add(v)\n"
    "    idx.append(i)\n"
    "print('C', count, len(evens), len(odds), sum(idx), len(seen), v)"
)


def _loop_line(output: str) -> str:
    rows = [line for line in output.splitlines() if "for " in line]
    assert rows, output
    return rows[0]


def test_a_nested_append_loop_restores_and_follows_an_edit(nb_runner):
    nb_runner.create_notebook([SETUP, GROUPS.format(n=3000), NESTED, "print('TOTAL', sum(deltas))"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = nb_runner.get_output(3)
    assert "N 9000 27009000 (2999, 3000, 3001) 3001" in cold, cold

    nb_runner.run_cells([3, 4])
    warm = nb_runner.get_output(3)
    assert "N 9000 27009000 (2999, 3000, 3001) 3001" in warm, warm
    assert _loop_line(warm).strip().startswith("CACHED:"), warm
    assert "TOTAL 27009000" in nb_runner.get_output(4)

    nb_runner.set_cell_source(2, GROUPS.format(n=2000))
    nb_runner.run_cells([2, 3, 4])
    edited = nb_runner.get_output(3)
    assert "N 6000 12006000 (1999, 2000, 2001) 2001" in edited, edited
    assert not _loop_line(edited).strip().startswith("CACHED:"), edited
    assert "TOTAL 12006000" in nb_runner.get_output(4)


def test_a_continue_loop_filling_four_containers_restores(nb_runner):
    nb_runner.create_notebook([SETUP, GROUPS.format(n=300), CONTINUE])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = nb_runner.get_output(3)
    assert "C 200 300 300 30000 301 301" in cold, cold

    nb_runner.run_cell(3)
    warm = nb_runner.get_output(3)
    assert "C 200 300 300 30000 301 301" in warm, warm
    assert _loop_line(warm).strip().startswith("CACHED:"), warm


@pytest.mark.parametrize(
    "seed, expect",
    [
        # Filled in an earlier cell (which cash replays before the loop).
        pytest.param("", "L 6 [0, 2, 4, 0, 2, 4]", id="seeded-in-an-earlier-cell"),
        pytest.param("out = [7]\n", "L 4 [7, 0, 2, 4]", id="seeded-non-empty"),
    ],
)
def test_a_list_not_seeded_empty_in_the_cell_re_runs(nb_runner, seed, expect):
    loop = f"{seed}for i in range(6):\n    if i % 2:\n        continue\n    out.append(i)\nprint('L', len(out), out)"
    nb_runner.create_notebook([SETUP, "out = [0, 2, 4]", loop])
    nb_runner.start_kernel()
    nb_runner.run_all()
    for _ in range(2):
        output = nb_runner.get_output(3)
        assert expect in output, output
        assert "In-place mutation on: out" in _loop_line(output), output
        nb_runner.run_cell(3)


def test_a_list_another_name_shares_stays_shared(nb_runner):
    loop = (
        "out = []\nview = out\n"
        "for i in range(6):\n    if i % 2:\n        continue\n    out.append(i)\n"
        "print('V', view, view is out)"
    )
    nb_runner.create_notebook([SETUP, loop, "out.append(99)\nprint('W', view)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    for _ in range(2):
        assert "V [0, 2, 4] True" in nb_runner.get_output(2), nb_runner.get_output(2)
        assert "W [0, 2, 4, 99]" in nb_runner.get_output(3), nb_runner.get_output(3)
        nb_runner.run_cells([2, 3])


def test_a_list_appended_into_the_container_keeps_its_identity(nb_runner):
    loop = (
        "rows = []\nrow = [0]\n"
        "for i in range(6):\n    if i % 2:\n        continue\n    rows.append(row)\n"
        "row.append(5)\nprint('R', rows[0] is row, rows[0])"
    )
    nb_runner.create_notebook([SETUP, loop])
    nb_runner.start_kernel()
    nb_runner.run_all()
    for _ in range(2):
        assert "R True [0, 5]" in nb_runner.get_output(2), nb_runner.get_output(2)
        nb_runner.run_cell(2)


def test_a_helper_changing_each_row_in_place_is_kept_on_a_hit(nb_runner):
    """``normalize(r)`` changes every row of ``rows``, not only the last one
    the loop variable ends on: a hit put back only that one, and the rows
    after a restart read as they were before the loop."""
    nb_runner.create_notebook(
        [
            SETUP + "\ncash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)",
            "def normalize(r):\n    r[0] = r[0] * 10\nrows = [[i, i] for i in range(400)]",
            "totals = []\nfor r in rows:\n    if r[1] < 0:\n        continue\n    normalize(r)\n"
            "    totals.append(r[0] + r[1])\nprint('TOTALS', totals[:3])",
            "print('ROWS', rows[:3], rows[-1] is r)",
        ]
    )
    nb_runner.start_kernel()
    for restart in (False, False, True):
        if restart:
            nb_runner.restart()
        nb_runner.run_all()
        assert "TOTALS [0, 11, 22]" in nb_runner.get_output(3)
        assert "ROWS [[0, 0], [10, 1], [20, 2]] True" in nb_runner.get_output(4), nb_runner.get_output(4)
