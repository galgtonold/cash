"""Sub-calls are grouped by call SITE, not by callee (CAS-243 task 9).

A call-unit ("``# @cash:cache-calls``") intercepted call is a cache line of
its own, keyed by where it's called from -- the same callee invoked from two
different statements (or twice in one statement) can have completely
different hit histories and cache keys. Grouping by ``func_name`` alone
(what the pre-existing ``DecoratorCallGroup`` does for hand-decorated calls)
would silently merge those two independent lines and hide exactly what a
reader opening the drawer is trying to debug.

This module also covers requirement 2: a call event raised inside a ``for``
loop body must inherit the loop's ``loop_header`` / ``loop_header_chain`` /
``body_index_chain`` stamps, or the badge view-builder has no way to nest it
under the loop and renders it as a sibling instead.
"""
from __future__ import annotations

import pytest
from traitlets.config import Configurable
from unittest.mock import MagicMock

from cash.core import Cash
from cash.backends import InMemoryBackend
from cash.notebook.ipython.magics import CashMagics
from cash.notebook.badge_renderer.view import (
    BadgeStatus,
    SubUnitGroup,
    build_sub_unit_groups,
)
from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.badge_renderer.renderers.html import render_html
from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.control_structures.for_handler import (
    _stamp_call_events_body_index,
    _stamp_call_events_loop_header,
)


def _event(source, occ, hit, key="call:abcdef0123456789", **extra):
    e = {
        "call_source": source, "occurrence_index": occ, "cache_hit": hit,
        "cache_key": key, "execution_time": 0.5, "time_saved": 0.5,
        "intercepted": True, "func_name": "compute",
    }
    e.update(extra)
    return e


# --------------------------------------------------------- build_sub_unit_groups

def test_two_sites_calling_the_same_function_are_two_groups():
    """The property this task exists for: SAME callee, DIFFERENT call sites.

    Mutation that breaks this: group key `e.get("func_name")` instead of
    `(e.get("call_source"), e.get("occurrence_index"))` -- a naive
    "does a sub-call row render" test would still pass under that mutation
    since both events are still func_name=="compute"; this one would not
    (len(groups) would collapse to 1).
    """
    groups = build_sub_unit_groups([
        _event("compute(x)", 0, False),
        _event("compute(y)", 0, True),
    ])
    assert len(groups) == 2
    assert {g.call_source for g in groups} == {"compute(x)", "compute(y)"}


def test_repeated_calls_at_one_site_condense():
    groups = build_sub_unit_groups([_event("compute(x)", 0, True) for _ in range(100)])
    assert len(groups) == 1
    assert len(groups[0].calls) == 100
    assert groups[0].condensed is True


def test_condense_threshold_boundary():
    """Exactly at the threshold stays uncondensed; one more condenses."""
    at_threshold = build_sub_unit_groups([_event("compute(x)", 0, True) for _ in range(3)])
    assert at_threshold[0].condensed is False
    over_threshold = build_sub_unit_groups([_event("compute(x)", 0, True) for _ in range(4)])
    assert over_threshold[0].condensed is True


def test_key_prefix_is_exposed_for_debugging():
    groups = build_sub_unit_groups([_event("compute(x)", 0, False)])
    assert groups[0].key_prefix == "call:abcdef01"


def test_same_call_source_different_occurrence_index_are_two_groups():
    """Two textually-identical calls at different AST positions (e.g. two
    ``compute(x)`` literals in the same statement) must stay separate too --
    call_source alone is not a unique site, occurrence_index disambiguates."""
    groups = build_sub_unit_groups([
        _event("compute(x)", 0, True),
        _event("compute(x)", 1, False),
    ])
    assert len(groups) == 2
    assert {g.occurrence_index for g in groups} == {0, 1}


def test_non_intercepted_events_are_excluded():
    """A hand-decorated ``@cash.cache`` call has no call site to group by --
    it must not leak into sub_units even if it happens to carry call_source."""
    groups = build_sub_unit_groups([_event("compute(x)", 0, True, intercepted=False)])
    assert groups == []


def test_miss_reason_surfaces_when_present():
    groups = build_sub_unit_groups([_event("compute(x)", 0, False, miss_reason="no prior cache entry")])
    assert groups[0].miss_reason == "no prior cache entry"


def test_miss_reason_absent_is_none_not_fabricated():
    """CallUnit does not emit miss_reason today -- must read as None, never a
    made-up string."""
    groups = build_sub_unit_groups([_event("compute(x)", 0, False)])
    assert groups[0].miss_reason is None


def test_malformed_events_do_not_raise():
    """The badge must never blow up on a legacy/malformed event dict."""
    groups = build_sub_unit_groups([
        {"intercepted": True},  # no call_source/occurrence_index/cache_hit at all
        "not even a dict",
        None,
    ])
    assert isinstance(groups, list)


# --------------------------------------------------------- view_builder wiring

def _metrics_two_sites():
    return [{
        "status": "COMPUTED",
        "code": "out = compute(x) + compute(y)",
        "total_time": 1.0,
        "evaluated_vars": ["out"],
        "decorator_calls": [
            _event("compute(x)", 0, False),
            _event("compute(y)", 1, True),
        ],
        "is_upstream": False,
    }]


def test_view_builder_populates_row_sub_units_grouped_by_site():
    badge = build_interactive_badge(_metrics_two_sites())
    rows = [
        item for section in badge.sections for item in section.items
        if hasattr(item, "sub_units")
    ]
    assert rows, "no StatementRow found in the built badge"
    row = rows[0]
    assert len(row.sub_units) == 2
    assert {g.call_source for g in row.sub_units} == {"compute(x)", "compute(y)"}
    assert all(isinstance(g, SubUnitGroup) for g in row.sub_units)


def test_html_badge_shows_both_call_sites_separately():
    """Both call sites get their OWN drawer entry, not the code echo alone.

    ``compute(x)``/``compute(y)`` also appear in the statement's own code
    text regardless of sub-unit wiring, so asserting on those alone would
    pass even with sub_units wholly absent (verified: the drawer-removal
    mutation below only trips the ``Sub-calls``/``cash-subunit`` count
    checks). The count assertion is the one that actually exercises
    per-site rendering.
    """
    html = render_html(build_interactive_badge(_metrics_two_sites()))
    assert "compute(x)" in html
    assert "compute(y)" in html
    assert "Sub-calls" in html
    assert html.count("cash-subunit") == 2, "expected one drawer entry per call site"


def test_text_badge_shows_both_call_sites_as_separate_lines():
    text = render_text(build_interactive_badge(_metrics_two_sites()))
    assert "sub-call compute(x)" in text, text
    assert "sub-call compute(y)" in text, text


def test_no_sub_units_means_no_sub_call_section():
    """Positive control: an ordinary statement with no intercepted calls must
    not grow a Sub-calls section."""
    metrics = [{
        "status": "COMPUTED", "code": "x = 1", "total_time": 0.01,
        "evaluated_vars": ["x"], "is_upstream": False,
    }]
    html = render_html(build_interactive_badge(metrics))
    text = render_text(build_interactive_badge(metrics))
    assert "Sub-calls" not in html
    assert "sub-call" not in text


# --------------------------------------------------------- loop-nesting stamps

def test_stamp_call_events_loop_header_propagates_to_events():
    m = {"decorator_calls": [_event("compute(x)", 0, True)]}
    _stamp_call_events_loop_header(m, "for x in items:")
    event = m["decorator_calls"][0]
    assert event["loop_header"] == "for x in items:"
    assert event["loop_header_chain"] == ["for x in items:"]


def test_stamp_call_events_loop_header_prepends_for_nesting():
    """Outer loop's header must end up FIRST in the chain (outermost-first),
    matching the enclosing metric's own chain convention exactly."""
    m = {"decorator_calls": [_event("compute(x)", 0, True)]}
    _stamp_call_events_loop_header(m, "for y in inner:")   # inner runs first
    _stamp_call_events_loop_header(m, "for x in outer:")   # then outer prepends
    event = m["decorator_calls"][0]
    assert event["loop_header_chain"] == ["for x in outer:", "for y in inner:"]
    # first-writer-wins for the scalar field, same as the metric-level rule
    assert event["loop_header"] == "for y in inner:"


def test_stamp_call_events_body_index_propagates_to_events():
    m = {"decorator_calls": [_event("compute(x)", 0, True)]}
    _stamp_call_events_body_index(m, 2)
    assert m["decorator_calls"][0]["body_index_chain"] == [2]


def test_stamp_call_events_skips_non_dict_events():
    """Must never raise on a malformed event -- badge plumbing is cosmetic."""
    m = {"decorator_calls": ["not a dict", None]}
    _stamp_call_events_loop_header(m, "for x in items:")
    _stamp_call_events_body_index(m, 0)  # no exception


def test_stamp_call_events_no_decorator_calls_key_is_a_noop():
    m = {}
    _stamp_call_events_loop_header(m, "for x in items:")
    _stamp_call_events_body_index(m, 0)
    assert m == {}


# --------------------------------------------------------- real pipeline (e2e)

class MockShell(Configurable):
    """Same mock shell ``test_single_unit_caching.py`` uses for real per-
    iteration for-loop tests -- runs the actual production pipeline
    (``CashMagics`` -> ``StatementProcessor`` -> ``ForLoopHandler`` ->
    ``CallCache``/``CallUnit``), just with IPython's shell mocked out.
    """

    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


def test_real_for_loop_stamps_call_events_with_loop_header(magics_fixture):
    """End-to-end: an intercepted call raised inside a REAL for-loop body
    (through the actual processor/for_handler/call_cache pipeline, not a
    hand-built dict) comes out with the same loop stamps its enclosing
    body-statement metric carries.

    ``magics._last_cell_metrics`` (used by ``%cash_status``) turned out to be
    the wrong capture point -- ``_update_last_cell_metrics`` narrows each
    metric down to 6 display fields and drops ``decorator_calls``/
    ``loop_header`` entirely (confirmed by inspection: every field this test
    needs was always ``None`` there, for ANY statement, regardless of this
    task's changes). The badge-render call is what receives the full raw
    ``ProcessResult`` dicts, so this spies on
    ``CashMagics._render_interactive_badge`` and reads its ``metrics_list``
    argument instead.

    This is requirement 2's guarantee at the point that matters -- the unit
    tests above prove the stamping helper is correct in isolation; this
    proves for_handler actually calls it for real intercepted events, not
    just for the metric dict itself.
    """
    # Historically NOT ``out.append(compute(t))`` -- that single-Expr-body
    # shape used to match a dispatch (``cacheable_accumulator_loop``, called
    # directly from ``control_structures/processor.py``) that routed the
    # WHOLE loop through single-unit caching REGARDLESS of size, never
    # reaching per-iteration decomposition at all. CAS-259 (2026-07-31) fixed
    # that: the shape is now consulted only from INSIDE
    # ``ForLoopHandler``'s cost-based single-unit branch, so a small loop
    # like this one (3 items, well under the ~50-iteration threshold) always
    # decomposes per-iteration regardless of body shape -- the append form
    # would produce ``__iteration_context__`` metrics here too now. Kept as a
    # subscript-assignment body (``results[t] = compute(t)``) anyway, since
    # it's a genuinely different ``ast.Assign`` shape from the bare
    # ``Expr(Call)`` append form and there's no reason to narrow coverage.
    #
    # ``compute`` must also clear CallUnit's own ``_COST_FLOOR_S`` (10ms,
    # call_unit.py) or the call is never recorded at all regardless of
    # whether it was intercepted (confirmed: without the sleep, zero events
    # were logged even though the directive engaged with no warning).
    magics_obj, shell, backend = magics_fixture
    all_metrics = _run_real_for_loop_and_capture_metrics(magics_obj, shell, _LOOP_CODE)
    assert shell.user_ns['results'] == {1: 2, 2: 3, 3: 4}

    body_stmts = [m for m in all_metrics if '# __iteration_context__:' in m.get('code', '')]
    assert body_stmts, "expected per-iteration body metrics"

    stamped_events = [
        (m, e)
        for m in body_stmts
        for e in (m.get('decorator_calls') or [])
        if e.get('intercepted')
    ]
    assert stamped_events, (
        "no intercepted call events were recorded -- either cache-calls did not "
        "engage or the drain wiring regressed"
    )
    for owner, e in stamped_events:
        assert e.get('loop_header'), f"event missing loop_header: {e}"
        assert e.get('loop_header_chain'), f"event missing loop_header_chain: {e}"
        assert e.get('body_index_chain'), f"event missing body_index_chain: {e}"
        # Must match the ENCLOSING statement's own stamp exactly, not just be
        # present -- a wrong-but-nonempty value would pass a bare truthiness
        # check and still misplace the row in the badge.
        assert e['loop_header'] == owner['loop_header']
        assert e['loop_header_chain'] == owner['loop_header_chain']
        assert e['body_index_chain'] == owner.get('body_index_chain')


def _run_real_for_loop_and_capture_metrics(magics_obj, shell, code: str) -> list:
    """Run *code* through the real ``%%cash`` pipeline and return the raw
    ``metrics_list`` the LAST badge render saw (every statement's complete
    ``ProcessResult`` dict, decorator_calls/loop_header included).

    Shared by the stamping test above and the render-level test below --
    both need the exact same real, un-mocked pipeline output.
    """
    captured_metrics_lists: list[list] = []
    original_render = magics_obj._render_interactive_badge

    def _spy(metrics_list, *args, **kwargs):
        captured_metrics_lists.append(list(metrics_list))
        return original_render(metrics_list, *args, **kwargs)

    magics_obj._render_interactive_badge = _spy
    magics_obj.cash("", code.strip())
    assert captured_metrics_lists, "badge render was never called"
    return max(captured_metrics_lists, key=len)


# Same shape as the stamping test above -- see its comments for why NOT
# ``out.append(compute(t))`` (accumulator fast-path) and why ``compute``
# sleeps (CallUnit's cost floor).
_LOOP_CODE = """
import time
def compute(x):
    time.sleep(0.02)
    return x + 1

results = {}
# @cash:cache-calls
for t in [1, 2, 3]:
    results[t] = compute(t)
"""


def test_real_for_loop_renders_sub_calls_nested_under_the_loop(magics_fixture):
    """Render-level counterpart to the stamping test above.

    A loop-body statement renders through ``IterationRow`` via
    ``view_builder._iteration_row()``, NOT through
    ``_statement_row_from_metric`` -- the only function ``build_sub_unit_groups``
    was originally wired into. So stamping the raw events correctly (proven
    above by asserting on the metrics dicts) is necessary but NOT sufficient
    for the badge to show anything: a first version of this fix had the
    stamps present on every event and STILL rendered no "Sub-calls" section
    at all for a loop body, because ``IterationRow`` had no ``sub_units``
    field and nothing populated one. This test would have caught that --
    it calls the real renderers on the real pipeline's output, not just
    inspects the metrics dicts.
    """
    magics_obj, shell, backend = magics_fixture
    all_metrics = _run_real_for_loop_and_capture_metrics(magics_obj, shell, _LOOP_CODE)
    assert shell.user_ns['results'] == {1: 2, 2: 3, 3: 4}

    badge = build_interactive_badge(all_metrics)
    html = render_html(badge)
    text = render_text(badge)

    # HTML: ``_loop_stmt_sub_units_html`` (view_builder's aggregated
    # LoopStatement.sub_units) is only ever called from inside
    # ``_for_loop_group_html``'s per-body-statement block -- its markup
    # cannot be emitted from anywhere else in the renderer, so its presence
    # together with the loop-body row IS the nesting proof: this is not the
    # row-level ``<dt>Sub-calls</dt>`` drawer (that one only exists for a
    # bare ``StatementRow``, which a loop-body statement is not).
    assert "c3-loop-body" in html, "expected the loop to render at all"
    assert "c3-subunit-table" in html, (
        "expected the loop-nested Sub-calls block, not the row-level one"
    )
    assert "<dt>Sub-calls</dt>" not in html, (
        "loop-body sub-calls rendered via the WRONG (row-level) drawer path"
    )
    assert "Sub-calls" in html
    assert "compute(t)" in html

    # TEXT: a sub-call line must immediately follow the iteration's own
    # line, indented one step deeper than it -- proving it reads as nested
    # under that specific iteration/loop, not as a detached sibling line
    # dropped elsewhere in the output.
    lines = text.splitlines()
    sub_call_idxs = [i for i, ln in enumerate(lines) if "sub-call compute(t)" in ln]
    assert sub_call_idxs, f"no sub-call line in text output:\n{text}"
    for i in sub_call_idxs:
        assert i > 0, "a sub-call line must follow a statement line, not open the output"
        prev = lines[i - 1]
        sub_indent = len(lines[i]) - len(lines[i].lstrip(" "))
        prev_indent = len(prev) - len(prev.lstrip(" "))
        assert sub_indent > prev_indent, (
            f"sub-call line is not nested deeper than its statement line:\n"
            f"{prev!r}\n{lines[i]!r}"
        )
        assert "results[" in prev, (
            f"line before a sub-call line is not the owning iteration's row:\n{prev!r}"
        )
