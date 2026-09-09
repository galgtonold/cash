"""A ``fig.savefig()`` must not flush a figure that was rebuilt but never drawn.

Found while hunting the round-14 plot-cell report, and worse than it: asking for
an UNRELATED downstream cell after a kernel restart overwrote the user's chart
with a blank one. Measured on the reporter's own notebook, same file across one
session::

    after a genuine plot-cell run   purple=1502  saturated=1974
    after the touch (plot not run)  purple=1502  saturated=1974
    after asking for cell 13        purple=   0  saturated=   0   (mtime moved)

The PNG keeps the user's own 800x400 geometry, so it is not matplotlib's default
blank -- it is *their* figure, rebuilt and never filled. No error, no badge, and
a restart cannot undo it.

Why the existing defence missed it: ``statement_saves_current_pyplot_figure``
says the receiver-bound form "is defended by the carrier-history pass (its input
``fig`` is a tracked carrier)". That pass classifies carriers from the LIVE
object (``stateful_carrier_kind(user_ns.get(v))``, and ``…(None)`` is ``None``),
so after a restart ``fig`` is absent, the pass silently does nothing, and the
plan keeps ``plt.subplots()`` + ``fig.savefig()`` while dropping ``ax.plot()``.
The defence disappears exactly when reconstruction matters most -- which is also
why this never showed up in a no-restart reproduction.

These drive the guard on a synthetic trace, kernel-free, so the refusal is
deterministic; the empty namespace IS the post-restart condition.
"""
from __future__ import annotations

import types
import warnings

import pytest

from cash.exceptions import CashWarning
from cash.notebook.upstream.reexecution_planner import ReexecutionPlanner


def _planner(user_ns: dict) -> ReexecutionPlanner:
    vl = types.SimpleNamespace(shell=types.SimpleNamespace(user_ns=user_ns))
    return ReexecutionPlanner(vl, classifier=None, debug=False)


def _entry(stmt, outputs=(), inputs=()):
    # (stmt_code, outputs, inputs, input_hashes, output_hashes, extra)
    return (stmt, set(outputs), list(inputs), {}, {}, None)


def _plot_trace(chart="electronics.png"):
    """The reporter's cell shape: produce, fill, title, save.

    The fills carry ``ax`` among their OUTPUTS, which is how the real trace
    records an in-place mutation -- the carrier-history pass relies on the same
    thing ("Outputs catch the FLAT fill"). Modelling them with empty outputs
    makes the fixture unable to reproduce the path it is testing: the guard
    finds no fills, allows the write, and the test fails against working code.
    """
    return [
        _entry("fig, ax = plt.subplots(figsize=(8, 4))", outputs=("fig", "ax")),  # 0
        _entry("ax.plot(sub['month'], sub['margin_pct'])",
               outputs=("ax",), inputs=("ax", "sub")),                            # 1
        _entry("ax.set_title('Electronics margin %')",
               outputs=("ax",), inputs=("ax",)),                                  # 2
        _entry(f"fig.savefig('{chart}')", inputs=("fig",)),                        # 3
    ]


class TestPostRestart:
    """An EMPTY namespace is the post-restart condition that opens the hole."""

    def test_a_rebuilt_but_unfilled_figure_write_is_refused(self):
        planner = _planner({})           # nothing live: `fig` is gone
        trace = _plot_trace()
        # The dangerous plan: rebuild the figure [0] and save it [3], with the
        # statements that draw into it [1][2] left behind.
        with pytest.warns(CashWarning):
            kept, restored = planner._guard_unfilled_figure_writes(
                [0, 3], trace, [],
            )
        assert 3 not in kept, "the write was allowed to flush a blank figure"
        assert 0 in kept, "only the write should be dropped, not the producer"

    def test_a_coherently_rebuilt_figure_is_allowed(self):
        """The healthy plan: everything that fills the figure is scheduled."""
        planner = _planner({})
        with warnings.catch_warnings():
            warnings.simplefilter("error")     # any refusal here is a bug
            kept, _ = planner._guard_unfilled_figure_writes(
                [0, 1, 2, 3], _plot_trace(), [],
            )
        assert kept == [0, 1, 2, 3]

    def test_a_write_whose_figure_is_not_rebuilt_is_left_alone(self):
        """Producer not scheduled: nothing is inventing a blank figure here."""
        planner = _planner({})
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kept, _ = planner._guard_unfilled_figure_writes([3], _plot_trace(), [])
        assert kept == [3]

    def test_a_refused_write_is_dropped_from_the_restored_set_too(self):
        planner = _planner({})
        trace = _plot_trace()
        restored = [{"code": trace[3][0], "status": "CACHED"}]
        with pytest.warns(CashWarning):
            _, remaining = planner._guard_unfilled_figure_writes([0, 3], trace, restored)
        assert remaining == [], "a refused write must not linger as restored"


class TestOwnership:
    def test_a_live_carrier_is_left_to_the_carrier_history_pass(self):
        """When `fig` is live that pass owns the case; this one must not fire."""
        plt = pytest.importorskip("matplotlib.pyplot")
        import matplotlib
        matplotlib.use("Agg")
        fig, ax = plt.subplots()
        try:
            planner = _planner({"fig": fig, "ax": ax, "plt": plt})
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                kept, _ = planner._guard_unfilled_figure_writes(
                    [0, 3], _plot_trace(), [],
                )
            assert kept == [0, 3]
        finally:
            plt.close(fig)

    def test_module_level_plt_savefig_belongs_to_the_other_guard(self):
        """`plt.savefig` matches the same syntax but is owned elsewhere.

        Claiming it here would also treat the MODULE `plt` as a figure whose
        "fills" are every statement that touched it.
        """
        planner = _planner({})
        trace = [
            _entry("fig, ax = plt.subplots()", outputs=("fig", "ax")),
            _entry("ax.bar(names, totals)", outputs=("ax",), inputs=("ax",)),
            _entry("plt.savefig('out.png')", inputs=("plt",)),
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            kept, _ = planner._guard_unfilled_figure_writes([0, 2], trace, [])
        assert kept == [0, 2]


def test_the_guard_is_actually_wired_into_the_plan():
    """The tests above call the guard directly, so they stay green if it is
    never called. Verified: deleting the call site left all of them passing.

    A full pipeline test would have to stub the backward scan and five other
    collaborators, so this checks the one thing those tests cannot -- that
    ``_build_reexecution_plan`` still invokes it, after the carrier-history pass
    it backstops.
    """
    import inspect

    source = inspect.getsource(ReexecutionPlanner._build_reexecution_plan)
    assert "_guard_unfilled_figure_writes" in source, (
        "the guard is no longer called from the plan builder; the unit tests "
        "above would not have caught this"
    )
    assert (source.index("_complete_stateful_carrier_history")
            < source.index("_guard_unfilled_figure_writes")), (
        "the guard must run AFTER the carrier-history pass -- it exists to "
        "catch what that pass misses when the carrier is not live"
    )


class TestDetector:
    @pytest.mark.parametrize("code,expected", [
        ("fig.savefig('a.png')", "fig"),
        ("f2.savefig(p, dpi=200)", "f2"),
        ("x = 1", None),
        ("obj.figure.savefig('a.png')", None),   # not a bare receiver
        ("this is not python(", None),           # must not raise
    ])
    def test_receiver_detection(self, code, expected):
        assert ReexecutionPlanner._receiver_bound_figure_write(code) == expected
