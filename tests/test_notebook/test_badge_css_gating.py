"""Every badge inlines only the CSS it uses.

The stylesheet is duplicated into every saved notebook cell, and a simple
badge uses 44 of its 109 classes -- the rest style loops, control structures
and decorator detail it never renders.

Omitting a block that IS needed is an invisible failure: an unstyled loop
group that nobody notices until someone runs a loop. So the test is
equivalence, not a size check -- every class in the markup must resolve to
the same declarations it would under the full stylesheet.
"""
from __future__ import annotations

import re

import pytest

from cash.notebook.badge_renderer.renderers import html as H
from cash.notebook.badge_renderer.renderers.html import render_html
from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.cache_status import CacheStatus


def _split(html: str) -> tuple[str, str]:
    """(css, markup) for a rendered badge."""
    css = "".join(re.findall(r"<style>(.*?)</style>", html, re.S))
    return css, re.sub(r"<style>.*?</style>", "", html, flags=re.S)


def _declarations_for(css: str) -> dict[str, frozenset[str]]:
    """selector -> declarations, comments stripped, whitespace-normalised."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out: dict[str, frozenset[str]] = {}
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        key = re.sub(r"\s+", "", sel)
        decls = frozenset(d.strip().replace(" ", "") for d in body.split(";") if d.strip())
        out[key] = out.get(key, frozenset()) | decls
    return out


def _classes(markup: str) -> set[str]:
    return {c for group in re.findall(r'class="([^"]+)"', markup) for c in group.split()}


# Badge shapes that between them should exercise every feature block.
#
# The "loop" / "control" / "decorator" shapes below are not in the original
# brief: grouping is gated on fields the brief's simple/restored/mixed
# shapes never set (a loop needs "# __iteration_context__:" inside code, a
# control group needs a control_context key, a decorator group needs a
# decorator_calls key -- see view_builder.py), so without them
# test_every_feature_block_is_reachable fails no matter how the CSS is
# split. These three mirror the existing passing tests in
# test_badge_view_builder.py (test_for_loop_iterations_collapse_into_for_loop_group,
# test_control_group_metrics_collapse_into_control_group,
# test_decorator_calls_become_their_own_section_grouped_by_func).
SHAPES = {
    "simple": [
        {"code": "x = a + 1", "status": str(CacheStatus.COMPUTED), "total_time": 0.5},
    ],
    "restored": [
        {"code": "x = a + 1", "status": str(CacheStatus.RESTORED),
         "total_time": 0.01, "saved_time": 0.5, "source": "RAM", "restored_vars": ["x"]},
    ],
    "mixed": [
        {"code": "x = 1", "status": str(CacheStatus.COMPUTED), "total_time": 0.5},
        {"code": "y = 2", "status": str(CacheStatus.RESTORED), "total_time": 0.01,
         "saved_time": 0.3, "source": "DISK"},
        {"code": "z = 3", "status": str(CacheStatus.SKIPPED), "total_time": 0.0,
         "uncacheable_reasons": ["Too cheap to cache"]},
    ],
    "loop": [
        {"code": "# __iteration_context__:loop1\ny = x*2", "status": str(CacheStatus.COMPUTED),
         "total_time": 0.01, "loop_vars": {"x": 1}},
        {"code": "# __iteration_context__:loop1\ny = x*2", "status": str(CacheStatus.COMPUTED),
         "total_time": 0.02, "loop_vars": {"x": 2}},
    ],
    "control": [
        {"code": "z = 1", "status": str(CacheStatus.COMPUTED), "control_context": "ctx1",
         "branch_label": "if", "body_statements": ["if x:", "    z = 1"], "total_time": 0.01},
        {"code": "z += 1", "status": str(CacheStatus.COMPUTED), "control_context": "ctx1",
         "total_time": 0.005},
    ],
    "decorator": [
        {"code": "f()", "status": str(CacheStatus.COMPUTED), "total_time": 0.1,
         "decorator_calls": [
             {"func_name": "myf", "cache_hit": True, "execution_time": 0.001},
             {"func_name": "myf", "cache_hit": False, "execution_time": 0.05},
         ]},
    ],
}


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_gated_css_matches_the_full_stylesheet_for_every_class_used(shape):
    html = render_html(build_interactive_badge(SHAPES[shape]))
    css, markup = _split(html)

    gated = _declarations_for(css)
    full = _declarations_for(H._CSS)
    used = _classes(markup)

    for selector, decls in full.items():
        named = set(re.findall(r"\.([a-zA-Z0-9_-]+)", selector))
        if named and not (named & used):
            continue  # rule cannot apply to this badge
        assert selector in gated, (
            f"{shape}: rule {selector!r} applies to a class in the markup but "
            f"its block was not emitted"
        )
        assert gated[selector] == decls, (
            f"{shape}: rule {selector!r} was emitted with different declarations"
        )


def test_a_simple_badge_is_smaller_than_the_full_stylesheet():
    """The point of gating. Without this, emitting everything would pass above."""
    css, _ = _split(render_html(build_interactive_badge(SHAPES["simple"])))
    assert len(css) < len(H._CSS) * 0.8, (
        f"expected gating to drop unused blocks: {len(H._CSS)} -> {len(css)}"
    )


def test_every_feature_block_is_reachable():
    """A block no shape can trigger is dead weight or a broken class set."""
    triggered = set()
    for shape in SHAPES:
        _, markup = _split(render_html(build_interactive_badge(SHAPES[shape])))
        used = _classes(markup)
        for name, classes in H._FEATURE_CLASSES.items():
            if classes & used:
                triggered.add(name)
    unreachable = set(H._FEATURE_CLASSES) - triggered
    assert not unreachable, (
        f"no test shape renders these blocks, so gating them is unverified: "
        f"{sorted(unreachable)}. Add a shape that does, or leave those rules in core."
    )
