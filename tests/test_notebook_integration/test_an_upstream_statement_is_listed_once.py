"""An upstream statement the repair re-runs is listed once in the badge.

``^CACHED: models = {}`` and ``^CACHED: weekly = ...``
twice in a row in the Upstream list. The backward scan restored the statement
and a later planner pass scheduled it to run as well; the badge showed the
restore and the run's cache hit.
"""

PIN = (
    "cash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0, "
    "loop_split_max_iter_seconds=1.0, loop_split_min_remaining_seconds=0.0)\n"
)
SETUP = "import cash\n%load_ext cash\n%cash_badge print\n" + PIN + "%cash_on"


def _repeated_rows(out):
    rows = [line.strip() for line in out.splitlines() if line.strip().startswith("^")]
    return sorted({row.split("  (")[0] for row in rows if rows.count(row) > 1})


def test_a_rebound_variables_first_producer_is_listed_once(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            "src = sum(i*i for i in range(2_000_000))",
            "w = src + 1\nextra = 5",
            "res = w * 2\nprint(res)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "w = src + 1\nextra = 5\nw = w + extra")
    nb_runner.run_cells([4])
    out = nb_runner.get_output(4)
    assert str((sum(i * i for i in range(2_000_000)) + 6) * 2) in out, out
    assert _repeated_rows(out) == [], out


def test_an_accumulators_init_is_listed_once(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            "src = sum(i*i for i in range(2_000_000))",
            "acc = {}\nfor i in range(3):\n    acc[i] = src + i",
            "res = sum(acc.values())\nprint(res)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "acc = {}\nfor i in range(3):\n    acc[i] = src + 2*i")
    nb_runner.run_cells([4])
    out = nb_runner.get_output(4)
    assert str(3 * sum(i * i for i in range(2_000_000)) + 6) in out, out
    assert [r for r in _repeated_rows(out) if "acc = {}" in r] == [], out
