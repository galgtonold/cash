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
        decls = tuple(d.strip().replace(" ", "") for d in body.split(";") if d.strip())
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
    assert len(css) < len(H._CSS) * 0.75, f"expected a real reduction, got {len(H._CSS)} -> {len(css)}"


def _stylesheet_source() -> str:
    from importlib import resources

    return resources.files("cash.notebook.badge_renderer.renderers").joinpath("badge.css").read_text(encoding="utf-8")


def test_the_stylesheet_is_a_package_resource():
    """Shipped as a file beside the renderer, so the wheel must carry it."""
    assert ".c3-wrap" in _stylesheet_source()


def test_the_stylesheet_hard_codes_no_colour():
    """Colours come from the theme, so a palette change is a one-file change."""
    source = re.sub(r"/\*.*?\*/", "", _stylesheet_source(), flags=re.S)
    assert re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", source) == []


def test_every_token_the_stylesheet_reads_is_declared():
    used = set(re.findall(r"var\(--c3-([a-z0-9-]+)\)", H._CSS))
    declared = set(re.findall(r"--c3-([a-z0-9-]+):", H._CSS))
    assert used, "the stylesheet reads no tokens"
    assert used == declared


def test_the_tier_rack_comes_from_the_badge_being_rendered():
    """Two renders with different tier lists do not share a rack."""
    three = build_interactive_badge(SIMPLE, configured_tiers=("RAM", "REDIS", "DISK"))
    one = build_interactive_badge(SIMPLE, configured_tiers=("RAM",))
    assert render_html(three).count('class="c3-dot ') == 3
    assert render_html(one).count('class="c3-dot ') == 1
