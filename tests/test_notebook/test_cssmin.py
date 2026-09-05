"""The badge stylesheet ships in every saved notebook cell, so it is minified.

A blind ``re.sub`` over CSS corrupts quoted text. These tests pin the two
things that matter: the rule set is unchanged, and strings survive.
"""
from __future__ import annotations

import re

from cash.notebook.badge_renderer.renderers._cssmin import minify_css


def _rules(css: str):
    """Comment-free (selector, {declarations}) pairs, whitespace-normalised.

    Comments are stripped from BOTH sides before comparing. Without that the
    comparison reports false differences: an unminified selector match also
    captures the comment sitting above it, and the minified one cannot.
    """
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    out = []
    for sel, body in re.findall(r"([^{}]+)\{([^{}]*)\}", css):
        decls = frozenset(d.strip().replace(" ", "") for d in body.split(";") if d.strip())
        out.append((re.sub(r"\s+", "", sel), decls))
    return out


def test_rules_and_declarations_survive_minification():
    css = """
    /* a comment */
    .a > .b   {  color : red ;  padding: 0 2px; }
    .c:hover  {  margin: 0 !important; }
    """
    assert _rules(minify_css(css)) == _rules(css)


def test_quoted_text_is_never_touched():
    """Whitespace and separators inside strings are content, not syntax."""
    css = '.x::before { content: "a ;  b > c"; }'
    assert 'content:"a ;  b > c"' in minify_css(css)


def test_comments_are_removed():
    assert "/*" not in minify_css("/* gone */ .a { color: red; }")


def test_it_actually_shrinks_the_real_stylesheet():
    """Guards against a no-op minifier passing the equivalence tests."""
    from cash.notebook.badge_renderer.renderers import html as H

    full = H._CSS
    small = minify_css(full)
    assert _rules(small) == _rules(full), "minification changed the rule set"
    assert len(small) < len(full) * 0.75, (
        f"expected a meaningful reduction, got {len(full)} -> {len(small)}"
    )
