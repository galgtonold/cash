"""Validates the `upstream_trace` debugging harness (conftest) + cash.notebook._trace.

The simulation engine runs in the kernel subprocess and not all of its module
loggers are surfaced through IOPub, so we trace decisions to a file the kernel
inherits via CASH_TRACE_FILE. This test guards that the channel works end to end
so the harness stays trustworthy for future cache-correctness debugging.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def test_trace_captures_simulation_decisions(upstream_trace):
    # An upstream edit forces the simulation to flag + re-execute a producer.
    t = upstream_trace(
        ["x = 10", "y = x + 1\nprint(y)"],
        lambda r: (r.set_cell_source(1, "x = 20"), r.run_cell(2)),
    )
    # Harness plumbing: events are captured and split at run_all boundary.
    assert t.events("simulate_enter"), "no simulate_enter events captured on re-run"
    enter = t.events("simulate_enter")[0]
    assert set(enter) >= {"cell_idx", "reassigned", "mutated", "method_receivers"}
    # The simulation must have decided to re-execute the changed producer.
    assert any("x = 20" in s for s in t.scheduled()), t.scheduled()


def test_trace_is_noop_without_env(nb_runner):
    """Sanity: with no CASH_TRACE_FILE set, tracing is inert and runs normally."""
    nb_runner.create_notebook(["a = 1", "print(a + 1)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "2" in nb_runner.get_output(2)
