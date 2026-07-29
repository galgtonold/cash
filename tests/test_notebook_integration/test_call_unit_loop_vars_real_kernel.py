"""Real-kernel proof that `loop_vars` reaches an intercepted call (CAS-243).

The wiring under test: ``ForLoopHandler._process_one_iteration`` pushes the
current iteration's loop vars onto ``StatementProcessor``'s stack
(``loop_vars_scope``), and ``CallUnit._build_key`` reads them back through
``CallCache``'s ``loop_vars_provider`` at the moment an intercepted call is
actually invoked. Every other test of this feature either drives
``call_cache_key`` directly (proves the discrimination logic, not that the
values ever arrive) or runs in-process with the real IPython shell mocked out
(``tests/test_notebook/test_call_unit_loop_vars_wiring.py``). This file is
the one arm that goes through an actual Jupyter kernel end to end -- the
closest thing to what a user's notebook does.

The motivating shape: ``fetch_next(conn)`` inside a loop, where ``conn`` is a
bare ``Name`` whose lineage never moves (it is the same object all three
iterations) and there is no computed argument expression for ``arg_digests``
to hash. Without ``loop_vars``, all three iterations mint the identical key,
and the second and third iterations serve the first iteration's value --
wrong on the very first run, no pre-existing cache required.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

# Above CallUnit's cost floor (10ms), or the call is never stored at all and
# every assertion below would hold whether or not this feature works.
_SLEEP = 0.15


def _defs(log):
    return (
        "import time, pathlib\n"
        f"LOG = pathlib.Path(r'{log}')\n"
        "counter = {'n': 0}\n"
        "conn = 'db-connection'\n"
        "def fetch_next(conn):\n"
        "    counter['n'] += 1\n"
        "    with LOG.open('a') as fh:\n"
        "        fh.write('x\\n')\n"
        f"    time.sleep({_SLEEP})\n"
        "    return counter['n']\n"
    )


def _n(log):
    return len(log.read_text().splitlines()) if log.exists() else 0


# `results[t] = fetch_next(conn)` -- a subscript-assignment body, not
# `out.append(fetch_next(conn))`. The append shape is a bare `Expr(Call)`
# body and can be routed through the accumulator single-unit fast path
# instead of real per-iteration decomposition; the assignment shape forces
# the per-iteration path that pushes/pops `loop_vars`.
_LOOP = (
    "results = {}\n"
    "# @cash:cache-calls\n"
    "for t in [1, 2, 3]:\n"
    "    results[t] = fetch_next(conn)\n"
    "print('RESULTS', sorted(results.items()))\n"
)


def test_hidden_state_call_is_correct_on_the_first_run(nb_runner, tmp_path):
    """No pre-existing cache: each iteration must still get its own value.

    This is the CAS-243 bug reproduced live: ``conn``'s lineage is constant
    across all three iterations and ``fetch_next`` takes no other argument,
    so pre-``loop_vars`` all three iterations keyed identically and
    iterations 2/3 served iteration 1's cached ``1`` -- ``RESULTS
    [(1, 1), (2, 1), (3, 1)]`` instead of the correct
    ``[(1, 1), (2, 2), (3, 3)]``.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_defs(log), _LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()

    output = nb_runner.get_output(2)
    assert "RESULTS [(1, 1), (2, 2), (3, 3)]" in output, output
    assert _n(log) == 3, (
        f"expected exactly 3 real fetch_next() executions on the first run, got {_n(log)}"
    )


def test_hidden_state_call_hits_cache_on_rerun(nb_runner, tmp_path):
    """Re-running the loop cell must replay the same three values from cache
    -- zero further real calls -- proving the per-iteration keys are stable
    across runs, not just distinct within one.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([_defs(log), _LOOP])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 3

    nb_runner.run_cell(2)
    output = nb_runner.get_output(2)
    assert "RESULTS [(1, 1), (2, 2), (3, 3)]" in output, output
    assert _n(log) == 3, "compute() re-ran on an unchanged rerun of the loop cell"
