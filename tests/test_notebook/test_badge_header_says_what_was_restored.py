"""A cell that was mostly restored does not headline as EXECUTED.

Round 29, r29s4: "every badge headline says EXECUTED even when nearly
everything was restored". Their sweep cell read ``EXECUTED · 10.05s · saved
257.45s``; the only statements that ran were an import and ``sweep_rows =
[]``. Scanning 16 headlines they could not tell restored cells from
recomputed ones. The headline now says how many of each ran, and leads with
CACHED when the time it saved outweighs the time it spent.
"""
from cash.notebook.badge_renderer.renderers.html import render_html
from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.badge_renderer.view_builder import build_interactive_badge

MOSTLY_RESTORED = (
    [{"status": "COMPUTED", "code": "sweep_rows = []", "execution_time": 0.01, "total_time": 0.01}]
    + [{"status": "RESTORED", "code": f"r{i} = fit({i})", "saved_time": 20.0, "total_time": 0.3}
       for i in range(12)]
)
MOSTLY_RUN = [
    {"status": "COMPUTED", "code": "model = fit(X)", "execution_time": 30.0, "total_time": 30.0},
    {"status": "RESTORED", "code": "X = load()", "saved_time": 1.0, "total_time": 0.1},
]


def _header(metrics):
    return render_text(build_interactive_badge(metrics)).splitlines()[0]


def test_a_mostly_restored_cell_leads_with_cached_and_its_counts():
    header = _header(MOSTLY_RESTORED)
    assert header.startswith("[Cash] CACHED"), header
    assert "12 restored" in header and "1 ran" in header, header


def test_a_cell_that_mostly_ran_still_says_executed_with_its_counts():
    header = _header(MOSTLY_RUN)
    assert header.startswith("[Cash] EXECUTED"), header
    assert "1 ran" in header and "1 restored" in header, header


def test_a_cell_that_only_ran_is_unchanged():
    assert _header(MOSTLY_RUN[:1]) == "[Cash] EXECUTED (30.00s)"


def test_the_html_headline_agrees():
    html = render_html(build_interactive_badge(MOSTLY_RESTORED))
    assert "12 restored" in html and "1 ran" in html
