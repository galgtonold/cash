"""Upstream statements must come back CACHED, not re-execute.

This is the feature that makes cash worth more than a decorator: edit one cell,
run a downstream cell, and everything in between comes back from disk instead
of running again. Nothing asserted it.

Written while chasing tracker #75, which turned out to be a false alarm -- see
that ticket for why ``restore-dead`` proved nothing. These tests survived the
correction because the behaviour they pin is real and was genuinely unasserted:
nothing else in the suite checked that an upstream statement comes back CACHED
rather than running again.

They can fail, which is the part worth stating. Both go red under the
``upstream-dead`` mutation (``tests/mutations/``), so they are not the kind of
test that passes whatever the engine does. They do NOT go red under
``restore-dead`` -- and neither does anything else, because that mutation does
not change observable behaviour at all.

Two things these tests learned the hard way:

* Assert on ``^CACHED:``. The caret is what marks an upstream row; a test
  checking bare ``CACHED:`` passes under the mutation, because the current
  cell's own statements still hit.
* Do not assert that NO upstream row re-executes. A trivial statement
  (``mid = root * 10``) is cheaper to re-run than to restore, and cash
  correctly re-runs it. The claim worth pinning is that the EXPENSIVE upstream
  statement came back from cache.
"""
from __future__ import annotations

import re

import pytest

# Mirrors conftest's CASH_TEST_PIN_THRESHOLDS. Inlined rather than imported:
# nothing else imports it, and a relative import does not resolve here.
PIN_THRESHOLDS = (
    "cash.configure(call_cost_floor_seconds=0.0, "
    "min_execution_time_to_cache_seconds=0.0, "
    "loop_split_max_iter_seconds=1.0, "
    "loop_split_min_remaining_seconds=0.0)\n"
)

#: The expensive upstream statement. Real work rather than a sleep, and the
#: ASSIGNMENT itself is what costs -- so restoring the variable is what saves
#: the time, which is precisely the behaviour under test. A sleep on its own
#: line is a separate statement from the assignment, and then the row the
#: assertion lands on is the cheap one. Match rows on ``mid =`` rather than on
#: this text: the badge re-renders source from the AST, so ``3_000_000`` comes
#: back as ``3000000``.
EXPENSIVE = "mid = sum(i * i for i in range(3_000_000)) + root"


def _upstream_rows(output: str) -> list[str]:
    """Badge rows belonging to the UPSTREAM section (the ones marked ``^``)."""
    return [ln.strip() for ln in output.splitlines() if ln.strip().startswith("^")]


def _upstream_cached(output: str) -> bool:
    """True when at least one upstream row was restored.

    Strips ``NOT CACHED`` first, for the reason ``conftest.shows_cached`` gives:
    it contains ``CACHED`` and would otherwise read as a hit on a row that was
    never cached at all.
    """
    return "^CACHED:" in output.replace("^NOT CACHED:", "")


def _build(nb_runner):
    nb_runner.create_notebook([
        "%load_ext cash\n%cash_badge print\n" + PIN_THRESHOLDS + "%cash_on",
        "# @cash:persist\nroot = 2",
        "# @cash:persist\n" + EXPENSIVE,
        "# @cash:persist\nleaf = mid + 1\nprint('leaf =', leaf)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    return nb_runner


@pytest.mark.fresh_kernel
def test_the_expensive_upstream_statement_restores_rather_than_re_running(nb_runner):
    """After a restart, running only the leaf restores its upstream chain.

    A restart is what forces the virtual-restore path: memory is empty, so the
    upstream values cannot be trusted in place and must come from disk. Doing
    this with a warm kernel proves nothing, because the values are simply still
    there and no restore is needed.
    """
    r = _build(nb_runner)
    r.restart()
    r.run_cell(1)              # re-load the extension; the restart cleared it
    r.run_cell(4)              # only the leaf; the runner is 1-based

    out = r.get_raw_output(4)
    assert out.strip(), "no badge captured -- every assertion below would be vacuous"
    rows = _upstream_rows(out)
    assert rows, f"no upstream section at all:\n{out[:1200]}"

    assert _upstream_cached(out), (
        "no upstream statement restored from cache -- they all re-executed. "
        f"This is exactly what tracker #75's `restore-dead` mutation does.\n{out[:1200]}"
    )

    expensive = [r_ for r_ in rows if "mid =" in r_]
    assert expensive, f"the expensive upstream row is missing:\n{out[:1200]}"
    assert expensive[0].startswith("^CACHED:"), (
        f"the expensive upstream statement re-executed instead of restoring:\n"
        f"{expensive[0]}"
    )


@pytest.mark.fresh_kernel
def test_the_restored_upstream_row_reports_a_real_saving(nb_runner):
    """The label and the cost are separate claims; pin the one that matters.

    A row could say CACHED while the work happened anyway, so the label alone
    is not enough. Deliberately NOT a wall-clock comparison: every integration
    flake in this repo has been a timing measurement sitting near a threshold.
    Cash already publishes what the restore saved, and that number is zero if
    the work was not actually skipped.
    """
    r = _build(nb_runner)
    r.restart()
    r.run_cell(1)
    r.run_cell(4)

    out = r.get_raw_output(4)
    assert "leaf = 8999995500000500003" in out, (
        f"wrong value -- the chain did not rebuild correctly:\n{out[:800]}"
    )

    rows = [x for x in _upstream_rows(out) if "mid =" in x]
    assert rows, f"the expensive upstream row is missing:\n{out[:1200]}"
    row = rows[0]
    assert row.startswith("^CACHED:"), f"it re-executed:\n{row}"

    saved = re.search(r"saved ([0-9.]+)s", row)
    assert saved, f"a restored row reported no saving at all:\n{row}"
    assert float(saved.group(1)) > 0.0, (
        f"restored, but saved 0s -- the work was not actually skipped:\n{row}"
    )
