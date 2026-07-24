"""Structural tests for the v3 HTML renderer over the BadgeView IR.

Assertions target the stable ``c3-*`` class contract and the semantic
``data-kind`` / ``data-status`` attributes, not specific pixel values.
The visual surface is documented in
``design/notebook-badges/Badge.v3.jsx``.
"""

from __future__ import annotations

from cash.notebook.badge_renderer.renderers.html import render_html
from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.cache_status import CacheStatus


def test_empty_badge_produces_valid_v3_skeleton() -> None:
    html = render_html(build_interactive_badge([]))
    assert "<style>" in html
    assert "c3-card" in html and "c3-summary" in html and "c3-panel" in html
    assert 'data-kind="exec"' in html  # default = nothing cached
    assert "EXECUTED" in html


def test_restored_row_uses_cached_kind_and_saved_time() -> None:
    metrics = [{
        "code": "x = 1",
        "status": str(CacheStatus.RESTORED),
        "total_time": 0.001,
        "saved_time": 0.5,
        "source": "RAM",
        "restored_vars": ["x"],
    }]
    html = render_html(build_interactive_badge(metrics))
    assert 'data-kind="cached"' in html
    assert "CACHED" in html
    assert "↑0.50s" in html  # saved-time sub-label on time chip
    assert "c3-time-chip-cached" in html


def test_computed_row_uses_exec_kind_and_renders_tier_dots() -> None:
    metrics = [{
        "code": "y = expensive()",
        "status": str(CacheStatus.COMPUTED),
        "total_time": 1.23,
        "evaluated_vars": ["y"],
        "storage": ["RAM", "DISK"],
    }]
    html = render_html(build_interactive_badge(metrics))
    assert 'data-kind="exec"' in html
    assert "EXECUTED" in html
    assert "1.23s" in html
    assert "c3-dot-solid" in html  # both RAM and DISK present
    # Output vars are surfaced in the hover tooltip, not as a noisy prefix
    # on the code line.
    assert "<dt>Produced</dt>" in html
    assert ">y</dd>" in html


def test_uncacheable_row_renders_blocked_dots() -> None:
    metrics = [{
        "code": "print('x')",
        "status": str(CacheStatus.COMPUTED),
        "total_time": 0.1,
        "uncacheable_reasons": ["Side effect: print() (io)"],
    }]
    html = render_html(build_interactive_badge(metrics))
    assert "c3-dot-blocked" in html
    assert "c3-dots-warn" in html
    assert "Side effect: print()" in html  # tooltip text


def test_for_loop_with_multiple_body_stmts_emits_one_header_row() -> None:
    """The for-statement header should not duplicate per body statement.

    A 5-stmt × 4-iter loop must render as one `for x in [...]:` head row
    plus five body lines, not five duplicate `for x in [...]:` heads.
    """
    metrics = []
    for body_id, code in enumerate(["a = 1", "b = 2", "c = 3", "d = 4", "e = 5"]):
        for x in ("TSLA", "AAPL", "MSFT", "GOOG"):
            metrics.append({
                "code": f"# __iteration_context__: {body_id}_{x}\n{code}",
                "status": str(CacheStatus.COMPUTED),
                "total_time": 0.01,
                "loop_vars": {"x": x},
            })
    html = render_html(build_interactive_badge(metrics))
    import re
    head_rows = re.findall(r'class="c3-row c3-loop-head"', html)
    body_rows = re.findall(r'class="c3-rowx c3-loop-body"', html)
    assert len(head_rows) == 1, f"want 1 head row, got {len(head_rows)}"
    assert len(body_rows) == 5, f"want 5 body rows, got {len(body_rows)}"
    # Loop header shows the actual iteration values, not just a placeholder.
    assert "TSLA" in html and "GOOG" in html


def test_loop_iterations_render_as_collapsible_details_with_histogram() -> None:
    metrics = [
        {"code": "# __iteration_context__:loop1\ny = x*2",
         "status": str(CacheStatus.COMPUTED), "total_time": 0.01,
         "loop_vars": {"x": 1}},
        {"code": "# __iteration_context__:loop1\ny = x*2",
         "status": str(CacheStatus.COMPUTED), "total_time": 0.02,
         "loop_vars": {"x": 2}},
    ]
    html = render_html(build_interactive_badge(metrics))
    assert "c3-loop-head" in html
    assert "c3-loop-body" in html
    assert "c3-iter-strip" in html  # inline mini-histogram
    assert "<details" in html
    # Loop header is syntax-highlighted (for/in wrapped in c3-kw spans).
    assert ">for</span>" in html and ">in</span>" in html
    # Per-iteration drill-down is rendered under the body <details>
    assert "c3-iter-table" in html
    assert "c3-iter-row" in html


def test_skipped_bucket_renders_as_collapsible_with_count_and_saved() -> None:
    metrics = [
        {"code": "step1", "status": str(CacheStatus.SKIPPED),
         "is_upstream": True, "saved_time": 0.3},
        {"code": "step2", "status": str(CacheStatus.SKIPPED),
         "is_upstream": True, "saved_time": 0.2},
    ]
    html = render_html(build_interactive_badge(metrics))
    assert "c3-skipped" in html
    assert "2 upstream steps not re-run" in html
    assert "saved 0.50s" in html
    # Explanation tooltip on the disclosure makes the "why" discoverable.
    assert "cache already covered" in html


def test_upstream_rails_softened_by_section_cascade() -> None:
    """Upstream rows are de-emphasised by softening their status rail.

    This is a *structural* CSS cascade off the section-body wrapper, not a
    per-row class baked in by the view layer: the rows already live inside
    ``.c3-upstream-body`` (a section fact), so the softening belongs in the
    stylesheet, not on each ``StatementRow``.
    """
    metrics = [
        {"code": "setup()", "status": str(CacheStatus.RESTORED),
         "is_upstream": True, "saved_time": 0.3, "total_time": 0.0},
        {"code": "current()", "status": str(CacheStatus.COMPUTED), "total_time": 0.2},
    ]
    html = render_html(build_interactive_badge(metrics))
    # The upstream rows sit inside the section body wrapper...
    assert "c3-upstream-body" in html
    # ...and the rail softening is a cascade scoped to that wrapper.
    assert ".c3-upstream-body .c3-rail" in html


def test_decorator_section_renders_with_cache_tag() -> None:
    metrics = [{
        "code": "f()", "status": str(CacheStatus.COMPUTED), "total_time": 0.1,
        "decorator_calls": [
            {"func_name": "myf", "cache_hit": True, "execution_time": 0.001},
            {"func_name": "myf", "cache_hit": False, "execution_time": 0.05},
        ],
    }]
    html = render_html(build_interactive_badge(metrics))
    assert "c3-cache-tag" in html
    assert "DECORATOR CACHE" in html
    assert "myf" in html
    assert "HIT" in html and "MISS" in html


def test_condensed_decorator_group_renders_breakdown_strip() -> None:
    calls = [{"func_name": "f", "cache_hit": True, "execution_time": 0.001} for _ in range(4)]
    metrics = [{"code": "loop", "status": str(CacheStatus.COMPUTED), "total_time": 0.1,
                "decorator_calls": calls}]
    html = render_html(build_interactive_badge(metrics))
    # Condensed decorator group reuses the c3-loop-body <details> shell
    # so the same summary-toggle CSS applies.
    assert "c3-deco-strip" in html
    assert "c3-deco-bar" in html
    assert "all 4 cached" in html


def test_footer_always_renders_bug_report_link() -> None:
    metrics = [{"code": "x=1", "status": str(CacheStatus.COMPUTED), "total_time": 0.1}]
    html = render_html(build_interactive_badge(metrics))
    assert "c3-footer" in html
    assert "Report incorrect caching" in html


def test_overhead_renders_as_single_collapsed_row() -> None:
    """All sub-categories collapse into one row instead of N near-zero rows."""
    metrics = [{"code": "x=1", "status": str(CacheStatus.COMPUTED), "total_time": 1.0}]
    html = render_html(build_interactive_badge(
        metrics,
        timing_breakdown={"badge_init": 0.05, "upstream_check": 0.02},
        cell_total_time=1.1,
    ))
    body = html.split("</style>", 1)[1]   # strip <style> block (has the word 'OVERHEAD' in a comment)
    # Exactly one overhead row.
    assert body.count('class="c3-row c3-ovh"') == 1
    # Lowercase 'overhead' label, no shouting OVERHEAD banner.
    assert "overhead</span>" in body
    assert "OVERHEAD" not in body
    # Sub-category labels inline, self-describing (not "init"/"progress").
    assert "upstream check" in body and "badge setup" in body
    # No legacy decorative emojis in the rendered body.
    for icon in ("🏷", "📊", "↻", "⚙"):
        assert icon not in body, f"unexpected legacy emoji in overhead: {icon}"


def test_running_state_summary_when_step_info_provided() -> None:
    html = render_html(build_interactive_badge(
        [], status="RUNNING", current_step=2, total_steps=5, current_code="z = slow()",
    ))
    assert "PROCESSING" in html
    assert "(2/5)" in html


def test_syntax_highlighting_wraps_keywords() -> None:
    metrics = [{"code": "for i in range(10):", "status": str(CacheStatus.COMPUTED),
                "total_time": 0.1}]
    html = render_html(build_interactive_badge(metrics))
    assert 'class="c3-kw"' in html  # for, in are keywords
    assert 'class="c3-num"' in html  # 10 is a number


def test_function_changed_renders_as_notification_row() -> None:
    metrics = [{
        "code": "x = compute()", "status": "FUNCTION_CHANGED",
        "is_upstream": True, "changed_functions": ["compute"],
    }]
    html = render_html(build_interactive_badge(metrics))
    assert "c3-notif-pill" in html
    assert "changed" in html
    assert "compute" in html


def test_skipped_row_does_not_claim_storage() -> None:
    """SKIPPED ('not re-run') rows must not show a filled RAM dot — the
    value was never produced this run, so claiming it lives in RAM is a lie."""
    metrics = [
        {"code": "step1()", "status": str(CacheStatus.SKIPPED), "is_upstream": True,
         "saved_time": 0.3},
        {"code": "current()", "status": str(CacheStatus.COMPUTED), "total_time": 0.1},
    ]
    html = render_html(build_interactive_badge(metrics))
    body = html.split("</style>", 1)[1]
    # The skipped row's dots cell renders both dots as empty.
    skip_row = body[body.find("step1()"):body.find("step1()") + 800]
    assert 'c3-dot-solid' not in skip_row
    assert 'c3-dot-empty' in skip_row


def test_loop_head_carries_aggregate_drawer_and_body_uses_per_iter_drill() -> None:
    """Loop head click shows aggregate counts; loop body click shows the
    per-iteration drill-down (under the threshold) or the counts drawer
    (above it). Below threshold (2 iters), the body row uses the
    per-iter table rather than the summary drawer."""
    metrics = [
        {"code": "# __iteration_context__: a\ny = x*2",
         "status": str(CacheStatus.COMPUTED), "total_time": 0.05,
         "loop_vars": {"x": 1}},
        {"code": "# __iteration_context__: b\ny = x*2",
         "status": str(CacheStatus.COMPUTED), "total_time": 0.07,
         "loop_vars": {"x": 2}},
    ]
    html = render_html(build_interactive_badge(metrics))
    body = html.split("</style>", 1)[1]
    # Head row: condensed-summary drawer with counts.
    head_block = body[body.find("c3-loop-head"):body.find("c3-loop-body")]
    assert "c3-rowtip" in head_block
    assert "Iterations" in head_block
    # Body row (below the iter threshold): per-iter drill-down table, not
    # a c3-rowtip summary.
    body_block = body[body.find("c3-loop-body"):]
    assert "c3-iter-table" in body_block
    assert "c3-iter-row" in body_block


def test_loop_body_with_many_iterations_caps_drilldown() -> None:
    """Above the iter inline limit the drilldown shows the first N rows
    and a single ``… +M more iterations`` row, instead of cliff-falling
    to a totally different "condensed counts" drawer (which lost the
    per-iter view entirely)."""
    metrics = [
        {"code": f"# __iteration_context__: {i}\ny = x*2",
         "status": str(CacheStatus.COMPUTED), "total_time": 0.01,
         "loop_vars": {"x": i}}
        for i in range(40)
    ]
    html = render_html(build_interactive_badge(metrics))
    body = html.split("</style>", 1)[1]
    body_block = body[body.find("c3-loop-body"):]
    # Drilldown table is still present — capped, not replaced.
    assert "c3-iter-table" in body_block
    assert "c3-iter-more" in body_block
    assert "more iteration" in body_block


def test_rows_use_checkbox_hack_for_click_to_expand_no_js() -> None:
    """Every row uses the input+label checkbox-hack — bulletproof across
    every browser, no <details>/<summary> quirks possible.

    <details>/<summary> failed to toggle on click in the user's host,
    despite the spec saying it should. Switched to the canonical
    checkbox-hack: hidden <input type=checkbox> + <label for=id> + CSS
    sibling combinators (:checked ~ .c3-rowtip). Works since IE9.
    """
    metrics = [{"code": "y = expensive()", "status": str(CacheStatus.COMPUTED),
                "total_time": 1.0, "evaluated_vars": ["y"], "storage": ["RAM"]}]
    html = render_html(build_interactive_badge(metrics))

    # No JS at all.
    assert "<script" not in html
    assert "document.createElement" not in html

    # Hidden checkbox + label-for-id pattern.
    assert 'class="c3-rxtog"' in html
    assert '<input type="checkbox"' in html
    assert '<label class="c3-row"' in html and 'for="rx-' in html

    # Drawer is a sibling of the checkbox; revealed via :checked ~ .c3-rowtip.
    assert "c3-rowtip" in html
    assert ".c3-rxtog:checked ~ .c3-rowtip" in html


def test_scoped_scrollbar_styling_present() -> None:
    """We style scrollbars only on ancestors that contain a Cash badge."""
    html = render_html(build_interactive_badge([]))
    # :has() scoping — won't affect cells without our badge.
    assert ":has(.c3-wrap)::-webkit-scrollbar" in html
    assert "scrollbar-color:" in html


def test_each_row_has_pure_css_hover_tooltip() -> None:
    """Tooltip is a sibling div inside .c3-row, revealed via :hover CSS."""
    metrics = [{
        "code": "y = expensive()",
        "status": str(CacheStatus.COMPUTED),
        "total_time": 1.0,
        "evaluated_vars": ["y"],
        "storage": ["RAM"],
    }]
    html = render_html(build_interactive_badge(metrics))
    assert "c3-rowtip" in html
    assert "<dt>Produced</dt>" in html        # vars surface in the drawer
    assert "<dt>Storage</dt>" in html


def test_overhead_bars_share_cell_max_scale_with_user_rows() -> None:
    """Overhead and statement bars must use the same denominator.

    A 0.05s overhead row in a cell with a 5s statement should be a small
    bar; if overhead had its own scale it'd be misleadingly full-width.
    """
    metrics = [{"code": "x=1", "status": str(CacheStatus.COMPUTED), "total_time": 5.0}]
    html = render_html(build_interactive_badge(
        metrics,
        timing_breakdown={"badge_init": 0.05, "upstream_check": 0.02},
        cell_total_time=5.1,
    ))
    import re
    widths = [float(w) for w in re.findall(r"c3-tbar-fill[^>]*width:([0-9.]+)%", html)]
    # The 5.0s statement should be the widest bar; overhead bars are well below.
    assert max(widths) >= 99.0
    overhead_widths = [w for w in widths if w < 50.0]
    assert overhead_widths, "overhead bars are missing"
    assert max(overhead_widths) < 30.0, f"overhead bars too wide: {overhead_widths}"


def test_sqrt_scaling_keeps_small_rows_visible() -> None:
    """A trivial row in a cell dominated by one slow row should still
    have a non-zero bar — linear scaling would make it invisible."""
    metrics = [
        {"code": "slow = expensive()", "status": str(CacheStatus.COMPUTED), "total_time": 100.0},
        {"code": "fast = quick()", "status": str(CacheStatus.COMPUTED), "total_time": 0.5},
    ]
    html = render_html(build_interactive_badge(metrics))
    import re
    widths = [float(w) for w in re.findall(r"c3-tbar-fill[^>]*width:([0-9.]+)%", html)]
    widths.sort()
    # 0.5/100 = 0.5% linear, but sqrt(0.005)*100 = ~7% — visible.
    assert widths[0] > 5.0, f"small row got squashed: {widths}"
    assert widths[-1] >= 99.0


def test_bug_report_link_uses_important_to_beat_jupyter_anchor_style() -> None:
    """Jupyter classic sets a global anchor color that needs explicit override."""
    html = render_html(build_interactive_badge([]))
    assert "a.c3-bug" in html  # explicit anchor selector for higher specificity
    assert "!important" in html


def test_summary_includes_sparkline_when_current_rows_exist() -> None:
    metrics = [
        {"code": "a=1", "status": str(CacheStatus.COMPUTED), "total_time": 0.1},
        {"code": "b=2", "status": str(CacheStatus.RESTORED), "total_time": 0.001,
         "saved_time": 0.5},
    ]
    html = render_html(build_interactive_badge(metrics))
    assert "c3-summary-spark" in html
    assert "c3-spark-bar" in html
    # Both filter counters present.
    assert "c3-fchip-exec" in html
    assert "c3-fchip-cached" in html
