"""Adversarial probes, wave 6 (2026-07-02): @stateful marker.

 1. test_stateful_marker_forces_reexecution — a @stateful-marked function's
        call cell must re-execute on an unchanged run_all (plain-kernel
        semantics: the hidden counter increments).

(A %cash_clear probe was removed: no such magic exists — noted on CAS-32.)
"""

import pytest

pytestmark = [pytest.mark.timeout(120)]

MISS = "Executing (cache miss)"


def test_stateful_marker_forces_reexecution(nb_runner):
    nb_runner.create_notebook([
        "from cash.notebook.purity import stateful\n"
        "calls6 = []\n"
        "@stateful\n"
        "def next_id():\n"
        "    calls6.append(1)\n"
        "    return len(calls6)",
        "nid = next_id()\nprint('nid=', nid)",
    ])
    nb_runner.start_kernel()
    nb_runner.enable_debug()
    nb_runner.run_all()
    assert "nid= 1" in nb_runner.get_output(2)

    nb_runner.run_all()
    out = nb_runner.get_output(2)
    # Plain-kernel semantics for a stateful call on a full rerun: increments.
    # (Function cell re-runs too, resetting calls6 -> next_id() returns 1 again;
    # so the observable invariant is simply: no crash and nid consistent with a
    # genuine re-execution, i.e. 1 — the cell must NOT be skipped into printing
    # a stale replay marked as cached.)
    assert "nid= 1" in out, f"stateful rerun output wrong: {out!r}"
    raw = nb_runner.get_raw_output(2)
    assert MISS in raw, (
        "@stateful call cell was served from cache / skipped on rerun "
        "(marker promises re-execution)"
    )


