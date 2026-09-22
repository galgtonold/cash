"""A cached call that re-ran says which of its inputs moved.

Round 30, r30s4: after an upstream edit the parameter sweep re-ran all six
points. The badge said "Sub-calls: 0/6 hit sweep_point(mg, npc)" and
"@intercepted 0 of 6 cached" -- "no row says WHY the key changed (no
'changed:' note)". The tester guessed wrong about the cause and wrote it up
as a suspected bug; the re-run was right, and one word on the badge would
have said so.

A statement already gets this ("input changed: x, y", from the backward
scan). A call now gets the same, from the components of its own key: what
the site was keyed on last time against what it is keyed on now.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print\nimport time",
    "MIN_GENES = 200\nSWEEP = [1, 2]",
    "rows = list(range(400))",
    "def sweep_point(data, k):\n    time.sleep(0.4)\n    return sum(data) * k\n"
    "res = [sweep_point(rows, k) for k in SWEEP]\nprint('RES', res)",
]


def test_the_badge_names_the_input_that_moved(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "RES [79800, 159600]" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    # An edit upstream that really does change what the sweep reads.
    nb_runner.set_cell_source(3, "rows = list(range(401))")
    nb_runner.run_cell(3)
    nb_runner.run_cell(4)
    raw = nb_runner.get_raw_output(4)
    assert "rows" in raw and "changed" in raw, (
        "the sweep re-ran and the badge did not say what changed:\n" + raw)


def test_an_unchanged_re_run_says_nothing(nb_runner):
    """Control: a cell that hits has nothing to explain, and a first run is
    not 'changed' either -- the deliberate silence statements already keep."""
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.get_raw_output(4)
    assert "changed" not in first, first

    nb_runner.run_cell(4)
    assert "changed" not in nb_runner.get_raw_output(4), nb_runner.get_raw_output(4)
