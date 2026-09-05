"""The badge stylesheet ships minified.

It is inlined into every badge so a saved notebook renders standalone, which
duplicates it into every cell: measured at 354.9 KB across a twelve-cell
notebook, 67% of the file. Minifying is the part of that cash can take back
without needing to know what the document already contains.

Minified once at import, not at build time and not as a committed generated
file: 0.57ms against a 157ms `import cash`, and it keeps an editable dev
install byte-identical to what users receive.
"""
from __future__ import annotations

import re

from cash.notebook.badge_renderer.renderers import html as H
from cash.notebook.badge_renderer.renderers.html import render_html
from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.cache_status import CacheStatus


def _rules(css: str):
    """Comment-free (selector, {declarations}) pairs, whitespace-normalised.

    Comments are stripped from BOTH sides before comparing: an unminified
    selector match also captures the comment block above the rule, and the
    minified one cannot, which otherwise reports ~34 false differences.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out = []
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        decls = frozenset(d.strip().replace(" ", "") for d in body.split(";") if d.strip())
        out.append((re.sub(r"\s+", "", sel), decls))
    return out


def _emitted_css(metrics) -> str:
    html = render_html(build_interactive_badge(metrics))
    return "".join(re.findall(r"<style>(.*?)</style>", html, re.S))


SIMPLE = [{"code": "x = a + 1", "status": str(CacheStatus.COMPUTED), "total_time": 0.5}]


def test_the_emitted_stylesheet_is_minified():
    css = _emitted_css(SIMPLE)
    assert "/*" not in css, "comments should not ship in every badge"
    assert "\n" not in css, "expected collapsed whitespace"


def test_minification_changed_no_rule():
    """The gate. Byte savings are worthless if a declaration moved."""
    assert _rules(_emitted_css(SIMPLE)) == _rules(H._CSS)


def test_it_is_meaningfully_smaller():
    """Guards against a no-op: the assertions above pass on unminified CSS
    only if it happens to lack comments, which this stylesheet does not."""
    css = _emitted_css(SIMPLE)
    assert len(css) < len(H._CSS) * 0.75, (
        f"expected a real reduction, got {len(H._CSS)} -> {len(css)}"
    )
