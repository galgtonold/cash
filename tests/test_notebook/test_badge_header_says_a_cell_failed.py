"""A cell whose statement raised does not say EXECUTED in its header.

Round 25's r25s2 (and r25s1's repros): ``[Cash] EXECUTED (0.70s)`` above a row
reading ``ERROR: ints = array('i', ...)``. The header is the one line a reader
glances at; it read like the cell had run.
"""

from cash.notebook.badge_renderer.renderers.html import render_html
from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.badge_renderer.view import BadgeStatus
from cash.notebook.badge_renderer.view_builder import build_interactive_badge

METRICS = [
    {"status": "COMPUTED", "code": "x = load()", "execution_time": 0.5},
    {"status": "ERROR", "code": "y = x['missing']", "execution_time": 0.01, "error": "KeyError: 'missing'"},
]


def test_the_text_header_says_error():
    out = render_text(build_interactive_badge(METRICS))
    header = out.splitlines()[0]
    assert "ERROR" in header and "EXECUTED" not in header, out


def test_the_header_status_is_error():
    assert build_interactive_badge(METRICS).header.status is BadgeStatus.ERROR


def test_a_cell_without_an_error_is_unchanged():
    out = render_text(build_interactive_badge(METRICS[:1]))
    assert out.splitlines()[0].startswith("[Cash] EXECUTED"), out


def test_the_html_badge_renders():
    assert "ERROR" in render_html(build_interactive_badge(METRICS)).upper()
