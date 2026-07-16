"""Unit tests for ``decide_cacheability``.

Tests drive the merge function directly with synthetic inputs — no
``StatementProcessor`` is constructed.  This is the test-surface gain
that motivated extracting the function out of ``process_statement``.
"""
from __future__ import annotations

import ast

import pytest

from cash.notebook.annotations import CacheAnnotation
from cash.notebook.cacheability import analyze_statement
from cash.notebook.cacheability_decision import decide_cacheability, identity_coupled_reason


def _analysis(code: str):
    tree = ast.parse(code)
    return tree, analyze_statement(code, tree)


def _decide(
    code: str,
    *,
    inputs: set[str] | None = None,
    outputs: set[str] | None = None,
    annotation: CacheAnnotation | None = None,
    user_ns: dict | None = None,
    variable_lineage: dict | None = None,
    is_stateful_call=lambda _name: False,
    scan_forbidden=lambda _code, _user_ns, _tree: [],
):
    tree, analysis = _analysis(code)
    return decide_cacheability(
        code=code,
        tree=tree,
        inputs=inputs or set(),
        outputs=outputs or set(),
        annotation=annotation,
        analysis=analysis,
        user_ns=user_ns or {},
        variable_lineage=variable_lineage or {},
        is_stateful_call=is_stateful_call,
        scan_forbidden=scan_forbidden,
    )


class TestNoReasonsCacheable:
    def test_pure_assignment_is_cacheable(self):
        cacheable, reasons = _decide("y = x + 1", inputs={"x"}, outputs={"y"},
                                      user_ns={"x": 1}, variable_lineage={"x": "h"})
        assert cacheable is True
        assert reasons == []


class TestAnnotationNoCache:
    def test_no_cache_annotation_short_circuits(self):
        cacheable, reasons = _decide(
            "y = x + 1",
            inputs={"x"}, outputs={"y"},
            annotation=CacheAnnotation(no_cache=True),
            user_ns={"x": 1}, variable_lineage={"x": "h"},
        )
        assert cacheable is False
        assert reasons == ['@cash:no-cache annotation']

    def test_no_cache_wins_over_other_reasons(self):
        """no_cache fires first; mutation reason never gets computed."""
        cacheable, reasons = _decide(
            "lst.append(1)",
            inputs={"lst"}, outputs=set(),
            annotation=CacheAnnotation(no_cache=True),
            user_ns={"lst": []}, variable_lineage={"lst": "h"},
        )
        assert cacheable is False
        assert reasons == ['@cash:no-cache annotation']

    def test_persist_annotation_does_not_skip(self):
        cacheable, _ = _decide(
            "y = x + 1",
            inputs={"x"}, outputs={"y"},
            annotation=CacheAnnotation(persist=True),
            user_ns={"x": 1}, variable_lineage={"x": "h"},
        )
        assert cacheable is True


class TestForbiddenFunctions:
    def test_forbidden_reasons_returned(self):
        def fake_scan(code, user_ns, tree):
            return ["Calls forbidden function: input"]
        cacheable, reasons = _decide(
            "x = input()", inputs=set(), outputs={"x"},
            scan_forbidden=fake_scan,
        )
        assert cacheable is False
        assert reasons == ["Calls forbidden function: input"]

    def test_forbidden_scan_exception_does_not_crash(self):
        def bad_scan(*_a, **_k):
            raise TypeError("boom")
        cacheable, reasons = _decide(
            "y = x + 1", inputs={"x"}, outputs={"y"},
            user_ns={"x": 1}, variable_lineage={"x": "h"},
            scan_forbidden=bad_scan,
        )
        assert cacheable is True
        assert reasons == []


class TestStatefulCall:
    def test_stateful_call_short_circuits(self):
        cacheable, reasons = _decide(
            "y = step()", inputs=set(), outputs={"y"},
            is_stateful_call=lambda name: name == "step",
        )
        assert cacheable is False
        assert reasons == ["Calls @stateful function"]

    def test_first_stateful_match_wins(self):
        """Function returns as soon as one stateful call is found."""
        seen = []
        def lookup(name):
            seen.append(name)
            return name == "step"
        _decide(
            "y = step()", inputs=set(), outputs={"y"},
            is_stateful_call=lookup,
        )
        assert seen == ["step"]

    def test_stateful_lookup_exception_does_not_crash(self):
        def bad_lookup(_name):
            raise AttributeError("boom")
        cacheable, reasons = _decide(
            "y = foo()", inputs=set(), outputs={"y"},
            is_stateful_call=bad_lookup,
        )
        assert cacheable is True
        assert reasons == []


class TestMutationsAndSideEffects:
    def test_in_place_mutation_blocks_caching(self):
        cacheable, reasons = _decide(
            "lst.append(1)", inputs={"lst"}, outputs=set(),
            user_ns={"lst": []}, variable_lineage={"lst": "h"},
        )
        assert cacheable is False
        assert any("In-place mutation" in r for r in reasons)

    def test_mutation_on_output_is_fine(self):
        cacheable, _ = _decide(
            "lst = []\nlst.append(1)",
            inputs=set(), outputs={"lst"},
        )
        assert cacheable is True


class TestMissingLineage:
    def test_input_without_lineage_blocks_caching(self):
        cacheable, reasons = _decide(
            "y = x + 1", inputs={"x"}, outputs={"y"},
            user_ns={"x": 1}, variable_lineage={},  # no lineage for x
        )
        assert cacheable is False
        assert reasons == ['Input variable missing lineage']

    def test_input_not_in_user_ns_blocks_caching(self):
        cacheable, reasons = _decide(
            "y = x + 1", inputs={"x"}, outputs={"y"},
            user_ns={}, variable_lineage={},
        )
        assert cacheable is False
        assert reasons == ['Input variable missing lineage']

    def test_builtin_input_does_not_need_lineage(self):
        cacheable, _ = _decide(
            "y = len(z)", inputs={"len", "z"}, outputs={"y"},
            user_ns={"z": []}, variable_lineage={"z": "h"},
        )
        assert cacheable is True

    def test_get_ipython_skipped(self):
        cacheable, _ = _decide(
            "y = 1", inputs={"get_ipython"}, outputs={"y"},
        )
        assert cacheable is True

    def test_module_input_does_not_need_lineage(self):
        """Module-type inputs are lineage-exempt: their source is captured elsewhere."""
        import os as os_mod
        cacheable, _ = _decide(
            "y = os.getcwd()", inputs={"os"}, outputs={"y"},
            user_ns={"os": os_mod}, variable_lineage={},
        )
        assert cacheable is True

    def test_private_callable_input_does_not_need_lineage(self):
        """Private callables are lineage-exempt (bound methods etc.)."""
        class _Helper:
            def __call__(self): return 1
        cacheable, _ = _decide(
            "y = _h()", inputs={"_h"}, outputs={"y"},
            user_ns={"_h": _Helper()}, variable_lineage={},
        )
        assert cacheable is True


class TestReasonOrdering:
    """The first reason-source that triggers wins; later ones are not consulted."""

    def test_annotation_beats_forbidden(self):
        def scan(*_): return ["forbidden!"]
        _, reasons = _decide(
            "y = x", inputs=set(), outputs={"y"},
            annotation=CacheAnnotation(no_cache=True),
            scan_forbidden=scan,
        )
        assert reasons == ['@cash:no-cache annotation']

    def test_forbidden_beats_stateful(self):
        def scan(*_): return ["forbidden!"]
        _, reasons = _decide(
            "y = step()", inputs=set(), outputs={"y"},
            scan_forbidden=scan,
            is_stateful_call=lambda _: True,
        )
        assert reasons == ["forbidden!"]

    def test_stateful_beats_mutation(self):
        _, reasons = _decide(
            "lst.append(step())", inputs={"lst"}, outputs=set(),
            user_ns={"lst": []}, variable_lineage={"lst": "h"},
            is_stateful_call=lambda name: name == "step",
        )
        assert reasons == ["Calls @stateful function"]

    def test_mutation_beats_missing_lineage(self):
        _, reasons = _decide(
            "lst.append(1)", inputs={"lst"}, outputs=set(),
            user_ns={"lst": []}, variable_lineage={},  # no lineage AND mutation
        )
        assert any("In-place mutation" in r for r in reasons)
        assert 'Input variable missing lineage' not in reasons



class TestIdentityCoupledReason:
    """``identity_coupled_reason`` — the CAS-144 refusal of matplotlib Figure/Axes.

    These objects are only correct while they ARE the object pyplot's ``Gcf``
    registry points at.  Caching one makes the RAM tier deep-copy it, and
    ``Figure.__setstate__`` re-registers the COPY as pyplot's current figure --
    so ``plt.savefig()`` silently writes a blank image on the FIRST run.

    Figures here are built via ``matplotlib.figure.Figure`` directly rather than
    ``pyplot``: the predicate is a pure type check, so it needs no global state,
    and touching pyplot would make these tests share the process-wide ``Gcf``
    registry and backend with whatever xdist scheduled alongside them (observed
    flaking exactly that way).  The real identity behaviour is pinned by the
    integration tests against a live kernel, which is the trustworthy oracle.
    """

    def test_plain_values_are_not_coupled(self):
        for value in [1, "s", [1, 2, 3], {"a": 1}, None, (1, 2), {1, 2}]:
            assert identity_coupled_reason("v", value) is None

    def test_figure_is_refused(self):
        figure_mod = pytest.importorskip("matplotlib.figure")
        reason = identity_coupled_reason("fig", figure_mod.Figure())
        assert reason is not None
        assert "Figure" in reason and "fig" in reason

    def test_axes_is_refused_because_it_drags_its_figure(self):
        figure_mod = pytest.importorskip("matplotlib.figure")
        ax = figure_mod.Figure().add_subplot()
        reason = identity_coupled_reason("ax", ax)
        assert reason is not None
        assert "Axes" in reason

    def test_ndarray_of_axes_is_refused(self):
        """``fig, axes = plt.subplots(2, 2)`` binds an object-array of Axes."""
        figure_mod = pytest.importorskip("matplotlib.figure")
        axes = figure_mod.Figure().subplots(2, 2)
        assert identity_coupled_reason("axes", axes) is not None

    def test_1d_ndarray_of_axes_is_refused(self):
        """``fig, axes = plt.subplots(1, 2)`` binds a 1-D object-array of Axes."""
        figure_mod = pytest.importorskip("matplotlib.figure")
        axes = figure_mod.Figure().subplots(1, 2)
        assert identity_coupled_reason("axes", axes) is not None

    def test_dict_of_axes_is_refused(self):
        """CAS-155: ``plt.subplot_mosaic(...)`` returns ``dict[str, Axes]`` and
        typically binds ONLY the dict, so nothing bare-Figure/Axes co-occurs."""
        figure_mod = pytest.importorskip("matplotlib.figure")
        fig = figure_mod.Figure()
        axd = {'a': fig.add_subplot(2, 1, 1), 'b': fig.add_subplot(2, 1, 2)}
        assert identity_coupled_reason("axd", axd) is not None

    def test_nested_list_of_axes_is_refused(self):
        """CAS-155: a hand-built ``[[ax, ax], [ax, ax]]`` grid (no numpy)."""
        figure_mod = pytest.importorskip("matplotlib.figure")
        fig = figure_mod.Figure()
        grid = [[fig.add_subplot(2, 2, 1), fig.add_subplot(2, 2, 2)],
                [fig.add_subplot(2, 2, 3), fig.add_subplot(2, 2, 4)]]
        assert identity_coupled_reason("grid", grid) is not None

    def test_list_of_ndarray_rows_is_refused(self):
        """CAS-155: ``rows = list(axes)`` after ``subplots(2, 2)`` — a list of
        1-D ndarrays of Axes; the pre-fix one-level scan saw only the ndarrays."""
        figure_mod = pytest.importorskip("matplotlib.figure")
        rows = list(figure_mod.Figure().subplots(2, 2))
        assert identity_coupled_reason("rows", rows) is not None

    def test_container_scan_is_depth_bounded_and_cycle_safe(self):
        """The scan is depth-bounded: a self-referential container terminates
        (returns None rather than looping), and a coupled object within the cap
        is still found."""
        figure_mod = pytest.importorskip("matplotlib.figure")
        cyclic: list = []
        cyclic.append(cyclic)
        assert identity_coupled_reason("cyclic", cyclic) is None
        ax = figure_mod.Figure().add_subplot()
        assert identity_coupled_reason("wrapped", [[ax]]) is not None

    def test_other_matplotlib_artists_stay_cacheable(self):
        """Precision guard: do NOT blanket-ban matplotlib. Line2D is an Artist
        but is not identity-coupled to pyplot's globals, so it stays cacheable."""
        figure_mod = pytest.importorskip("matplotlib.figure")
        line = figure_mod.Figure().add_subplot().plot([1, 2], [3, 4])[0]
        assert identity_coupled_reason("line", line) is None

    def test_numeric_ndarray_is_not_scanned_elementwise(self):
        """A numeric array cannot hold an Axes; the dtype gate keeps it off the
        scan path entirely (and is why a big array costs nothing here)."""
        np = pytest.importorskip("numpy")
        assert identity_coupled_reason("arr", np.arange(100_000)) is None
