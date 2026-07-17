"""Unit tests for the CAS-187 orphaned-``plt.savefig()`` guard.

``plt.savefig(path)`` saves pyplot's process-global current figure; its only
variable input is the module ``plt``, so the value-lineage planner has no edge
to follow to the figure. When the re-execution plan schedules such a write while
the ``plt.subplots()`` that registered the current figure is NOT scheduled,
re-running the write makes ``plt.gcf()`` invent a blank figure and flush it over
the user's chart. These tests drive the guard directly on a synthetic
simulation trace so the refusal is deterministic and kernel-free.
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


@pytest.fixture
def fig_ax():
    plt = pytest.importorskip("matplotlib.pyplot")
    import matplotlib
    matplotlib.use("Agg")
    fig, ax = plt.subplots()
    yield fig, ax
    plt.close(fig)


# The canonical draw-on-ax / save-via-plt cell, split into three trace entries.
def _bar_trace(chart="out.png"):
    return [
        _entry("fig, ax = plt.subplots()", outputs=("fig", "ax")),         # 0 producer
        _entry("ax.bar(names, totals)", inputs=("ax", "names", "totals")),  # 1 fill
        _entry(f"plt.savefig('{chart}')", inputs=("plt",)),                 # 2 the write
    ]


class TestVulnerableShapeRefused:
    def test_orphaned_savefig_is_dropped_and_warns(self, fig_ax):
        fig, ax = fig_ax
        import matplotlib.pyplot as plt
        ns = {"plt": plt, "fig": fig, "ax": ax}
        trace = _bar_trace()
        planner = _planner(ns)

        # The vulnerable plan: the write [2] is scheduled, its producer [0] is not.
        with pytest.warns(CashWarning, match="CAS-187"):
            remaining, restored = planner._guard_global_figure_writes(
                [2], trace, [],
            )
        assert remaining == [], "the orphaned plt.savefig() must be dropped from the plan"

    def test_refused_write_is_also_stripped_from_restored_info(self, fig_ax):
        fig, ax = fig_ax
        import matplotlib.pyplot as plt
        ns = {"plt": plt, "fig": fig, "ax": ax}
        trace = _bar_trace("chart.png")
        planner = _planner(ns)

        restored = [{"code": "plt.savefig('chart.png')", "status": "RESTORED"},
                    {"code": "something else", "status": "RESTORED"}]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            remaining, restored_out = planner._guard_global_figure_writes([2], trace, restored)
        assert remaining == []
        assert all(i.get("code") != "plt.savefig('chart.png')" for i in restored_out)
        assert any(i.get("code") == "something else" for i in restored_out)

    def test_write_with_no_figure_producer_in_trace_is_refused(self, fig_ax):
        """A scheduled ``plt.savefig`` with no figure-producing statement anywhere
        before it cannot be verified to save a figure the user drew -> refuse."""
        fig, ax = fig_ax
        import matplotlib.pyplot as plt
        ns = {"plt": plt}
        trace = [_entry("plt.savefig('x.png')", inputs=("plt",))]
        planner = _planner(ns)
        with pytest.warns(CashWarning):
            remaining, _ = planner._guard_global_figure_writes([0], trace, [])
        assert remaining == []


class TestHealthyShapeUntouched:
    def test_producer_scheduled_alongside_write_is_allowed(self, fig_ax):
        """The healthy path: when the producer [0] is ALSO scheduled the figure is
        (re)built coherently, so the write must be left in the plan and NOT warn."""
        fig, ax = fig_ax
        import matplotlib.pyplot as plt
        ns = {"plt": plt, "fig": fig, "ax": ax}
        trace = _bar_trace()
        planner = _planner(ns)

        with warnings.catch_warnings():
            warnings.simplefilter("error", CashWarning)  # any CashWarning fails the test
            remaining, restored = planner._guard_global_figure_writes(
                [0, 1, 2], trace, [],
            )
        assert remaining == [0, 1, 2], "a coherently-rebuilt figure write must be kept"

    def test_fig_savefig_is_not_a_global_write(self, fig_ax):
        """Receiver-bound ``fig.savefig`` is defended by the carrier-history pass;
        this guard must not touch it even when its producer is unscheduled."""
        fig, ax = fig_ax
        import matplotlib.pyplot as plt
        ns = {"plt": plt, "fig": fig, "ax": ax}
        trace = [
            _entry("fig, ax = plt.subplots()", outputs=("fig", "ax")),
            _entry("fig.savefig('out.png')", inputs=("fig",)),
        ]
        planner = _planner(ns)
        with warnings.catch_warnings():
            warnings.simplefilter("error", CashWarning)
            remaining, _ = planner._guard_global_figure_writes([1], trace, [])
        assert remaining == [1], "fig.savefig must not be refused by the global-write guard"

    def test_empty_plan_is_a_noop(self):
        planner = _planner({})
        assert planner._guard_global_figure_writes([], [], []) == ([], [])

    def test_no_user_ns_falls_back_to_textual_and_still_guards(self):
        """With no live namespace the receiver check falls back to the ``plt``
        alias, and the figure producer is found textually via plt.subplots()."""
        trace = [
            _entry("fig, ax = plt.subplots()", outputs=("fig", "ax")),
            _entry("plt.savefig('x.png')", inputs=("plt",)),
        ]
        vl = types.SimpleNamespace(shell=types.SimpleNamespace(user_ns=None))
        planner = ReexecutionPlanner(vl, classifier=None, debug=False)
        # producer [0] not scheduled -> refuse the write [1]
        with pytest.warns(CashWarning):
            remaining, _ = planner._guard_global_figure_writes([1], trace, [])
        assert remaining == []
        # producer [0] scheduled -> allow
        with warnings.catch_warnings():
            warnings.simplefilter("error", CashWarning)
            remaining2, _ = planner._guard_global_figure_writes([0, 1], trace, [])
        assert remaining2 == [0, 1]


class TestDetectionHelper:
    def test_distinguishes_module_from_receiver_via_namespace(self, fig_ax):
        from cash.notebook.cacheability import statement_saves_current_pyplot_figure as f
        fig, ax = fig_ax
        import matplotlib.pyplot as plt
        ns = {"plt": plt, "fig": fig}
        assert f("plt.savefig('x.png')", ns) is True
        assert f("fig.savefig('x.png')", ns) is False
        # A variable named ``plt`` that is NOT the module must not be flagged.
        assert f("plt.savefig('x.png')", {"plt": fig}) is False
        # matplotlib.pyplot.savefig chain is flagged regardless of namespace.
        assert f("matplotlib.pyplot.savefig(p)", None) is True
