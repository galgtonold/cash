"""A fill the carrier pass schedules must be able to RUN.

Round-14 BLOCKING report, reproduced after eleven attempts by constructing the
condition rather than guessing the session history::

    UpstreamStateError: Upstream statement "ax.plot(sub['month'],
    sub['margin_pct'], color='purple', lw=..." failed during auto-re-execution:
    NameError: name 'sub' is not defined.

raised on a completely unrelated downstream cell, blocking five cells at once in
the reporter's session.

``_complete_stateful_carrier_history`` re-executes a carrier's establishing
history so a figure is never rebuilt from a subset of it: for ``fig.savefig()``
it schedules ``fig, ax = plt.subplots()`` plus every statement between that and
the write which touches a co-produced name — ``ax.plot(...)``, ``ax.set_title(...)``.

It schedules those fills without their own DATA inputs. ``ax.plot(sub[...])``
reads ``sub``, ``sub`` is not a carrier, and the backward scan never treated it
as a broken var, so nothing pulls its producer in. When ``sub`` is also absent
from the namespace the fill cannot run.

Why it took so long to reproduce: ``fig``, ``ax`` and ``sub`` come from the same
cell, so they are normally all present or all absent — and both extremes are
harmless. `fig` absent means the pass never fires (it classifies carriers from
the live object); `sub` present means the fill just runs. Only a state that
separates them reaches this, which is why every "restart and re-run" attempt
came back clean.
"""
from __future__ import annotations

import types

import pytest

from cash.notebook.upstream.reexecution_planner import ReexecutionPlanner


def _planner(user_ns: dict) -> ReexecutionPlanner:
    vl = types.SimpleNamespace(shell=types.SimpleNamespace(user_ns=user_ns))
    return ReexecutionPlanner(vl, classifier=None, debug=False)


def _entry(stmt, outputs=(), inputs=()):
    # (stmt_code, outputs, inputs, input_hashes, output_hashes, extra)
    return (stmt, set(outputs), list(inputs), {}, {}, None)


def _trace():
    """The reporter's cell, with `sub`'s producer ahead of the figure."""
    return [
        _entry("mm = monthly_margin(tx)", outputs=("mm",), inputs=("tx",)),        # 0
        _entry("sub = mm[mm['category'] == 'Electronics']",
               outputs=("sub",), inputs=("mm",)),                                  # 1
        _entry("fig, ax = plt.subplots(figsize=(8, 4))", outputs=("fig", "ax")),   # 2
        _entry("ax.plot(sub['month'], sub['margin_pct'], color='purple')",
               outputs=("ax",), inputs=("ax", "sub")),                             # 3
        _entry("ax.set_title('Electronics margin %')",
               outputs=("ax",), inputs=("ax",)),                                   # 4
        _entry("fig.savefig('electronics.png')", inputs=("fig",)),                 # 5
    ]


@pytest.fixture
def live_figure():
    """`fig` must be LIVE or the carrier pass never fires at all."""
    plt = pytest.importorskip("matplotlib.pyplot")
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plt.subplots()
    yield plt, fig, ax
    plt.close(fig)


def test_a_scheduled_fill_gets_the_producer_of_the_data_it_reads(live_figure):
    """`ax.plot(sub[...])` is useless without `sub = mm[...]`."""
    plt, fig, ax = live_figure
    # `sub` deliberately absent: this is the state that makes the fill unrunnable.
    planner = _planner({"plt": plt, "fig": fig, "ax": ax})

    kept, _ = planner._complete_stateful_carrier_history([5], _trace(), [])

    assert 3 in kept, "the fill was not scheduled at all -- fixture is wrong"
    assert 1 in kept, (
        "`ax.plot(sub[...])` was scheduled without `sub = mm[...]`, so it will "
        "raise NameError during upstream re-execution"
    )
    assert kept.index(1) < kept.index(3), "the producer must run before the fill"


def test_the_producers_chain_transitively(live_figure):
    """`sub` needs `mm`, which needs `tx` -- the whole chain has to come."""
    plt, fig, ax = live_figure
    planner = _planner({"plt": plt, "fig": fig, "ax": ax})

    kept, _ = planner._complete_stateful_carrier_history([5], _trace(), [])

    assert 0 in kept, "`mm = monthly_margin(tx)` was left behind"
    assert kept.index(0) < kept.index(1) < kept.index(3)


def test_nothing_extra_is_scheduled_when_no_fill_needs_it(live_figure):
    """A fill with no data inputs must not drag the notebook in behind it."""
    plt, fig, ax = live_figure
    planner = _planner({"plt": plt, "fig": fig, "ax": ax})
    trace = [
        _entry("data = load()", outputs=("data",)),                         # 0
        _entry("fig, ax = plt.subplots()", outputs=("fig", "ax")),          # 1
        _entry("ax.grid(True)", outputs=("ax",), inputs=("ax",)),           # 2
        _entry("fig.savefig('out.png')", inputs=("fig",)),                  # 3
    ]

    kept, _ = planner._complete_stateful_carrier_history([3], trace, [])

    assert 2 in kept and 1 in kept
    assert 0 not in kept, "`data = load()` is unrelated to the figure"


def test_a_file_writing_producer_is_never_dragged_in(live_figure):
    """The one producer this must NOT schedule.

    ``_schedule_file_write_statements`` runs BEFORE the carrier pass, so a
    producer added here has already bypassed its scope and repeatability gates.
    Re-firing an append duplicates a line on disk, and a kernel restart cannot
    undo that -- whereas the NameError it leaves behind is loud, names the
    variable, and is fixed by running the cell.
    """
    plt, fig, ax = live_figure
    planner = _planner({"plt": plt, "fig": fig, "ax": ax})
    trace = [
        _entry("sub = audit_and_load(tx)\nsub.to_csv('audit.log', mode='a')",
               outputs=("sub",), inputs=("tx",)),                            # 0
        _entry("fig, ax = plt.subplots()", outputs=("fig", "ax")),           # 1
        _entry("ax.plot(sub['x'])", outputs=("ax",), inputs=("ax", "sub")),  # 2
        _entry("fig.savefig('out.png')", inputs=("fig",)),                   # 3
    ]

    kept, _ = planner._complete_stateful_carrier_history([3], trace, [])

    assert 2 in kept, "the fill should still be scheduled"
    assert 0 not in kept, (
        "a producer that appends to a file was scheduled; re-running it "
        "duplicates the audit line permanently"
    )


def test_the_pass_is_still_inert_without_a_live_carrier():
    """Unchanged behaviour: no live `fig`, no carrier classification.

    Pinned because it is the OTHER half of this defect -- the post-restart case
    is handled by the unfilled-figure-write guard, not here, and the two must
    not quietly swap responsibilities.
    """
    planner = _planner({})
    kept, _ = planner._complete_stateful_carrier_history([5], _trace(), [])
    assert kept == [5]
