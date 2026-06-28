"""``# @cash: no-cache`` makes a statement behave as if cash weren't installed:
re-running ADVANCES state (like plain Jupyter), for both reassignment and
in-place mutation. Previously the in-place case was wrongly reset to its
cell-entry base (CAS-51).

Root cause (found with the upstream-trace harness): a no-cache statement still
bumps its var's runtime lineage, so pass 2 of the simulation flagged the var
stale (runtime lineage advanced past the simulation's) and re-executed its
producer -- resetting it. CAS-47's no-cache exclusion only covered the
stale-value guard's self-write sets, not the pass-2 lineage mismatch. The fix
drops no-cache-written vars from ``broken_vars`` before producer scheduling.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _last(out: str) -> str:
    return out.strip().splitlines()[-1].strip()


def test_nocache_inplace_mutation_advances(nb_runner):
    nb_runner.create_notebook([
        "log = []",
        "# @cash: no-cache\nlog.append(len(log))\nprint(len(log))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(2)) == "1"
    nb_runner.run_cell(2)
    assert _last(nb_runner.get_output(2)) == "2", nb_runner.get_output(2)


def test_nocache_reassignment_advances(nb_runner):
    nb_runner.create_notebook([
        "counter = 0",
        "# @cash: no-cache\ncounter = counter + 1\nprint(counter)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(2)) == "1"
    nb_runner.run_cell(2)
    assert _last(nb_runner.get_output(2)) == "2", nb_runner.get_output(2)


def test_cached_inplace_mutation_still_resets(nb_runner):
    """Control: WITHOUT no-cache, the run-from-start guarantee still resets the
    in-place mutation on an isolated re-run (the fix is scoped to no-cache)."""
    nb_runner.create_notebook([
        "log = []",
        "log.append(len(log))\nprint(len(log))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _last(nb_runner.get_output(2)) == "1"
    nb_runner.run_cell(2)
    assert _last(nb_runner.get_output(2)) == "1", nb_runner.get_output(2)


def test_nocache_inplace_does_not_reexecute_producer(upstream_trace):
    """The producer of a no-cache-mutated var must not be scheduled for re-run."""
    t = upstream_trace(
        ["log = []", "# @cash: no-cache\nlog.append(len(log))\nprint(len(log))"],
        lambda r: r.run_cell(2),
    )
    assert "log = []" not in t.scheduled(), t.scheduled()
    assert t.events("broken_drop_nocache"), "expected log dropped from broken_vars"
