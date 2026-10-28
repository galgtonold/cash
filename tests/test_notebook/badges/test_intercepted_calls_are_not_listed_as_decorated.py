"""The badge lists calls cash wrapped itself under their own heading.

A loop of four passes that each called ``pd.concat`` 2495 times read::

      sub-call pd.concat([df, x], ignore_index=False): 0/2495 hit, ...
    @cash.cache:
      concat() [intercepted]: 0/4 cached (2.040s)

The notebook decorated nothing, and ``0/4`` counted the passes while the
sub-call lines above it counted the calls.
"""

from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.badge_renderer.view import SectionKind
from cash.notebook.badge_renderer.view_builder import build_interactive_badge


def _event(name, source, *, intercepted=True, hit=False):
    return {
        "func_name": name,
        "cache_hit": hit,
        "execution_time": 0.5,
        "intercepted": intercepted,
        "cache_key": "call:x",
        "call_source": source,
        "occurrence_index": 0,
        "stored": True,
    }


def _metrics(*events, code="y = f(x)"):
    return [
        {"status": "COMPUTED", "code": code, "execution_time": 1.0, "total_time": 1.0, "decorator_calls": list(events)}
    ]


def test_an_intercepted_call_is_not_under_the_decorator_heading():
    out = render_text(build_interactive_badge(_metrics(_event("pandas.concat", "pd.concat([a, b])"))))
    assert "@cash.cache:" not in out, out
    assert "intercepted calls:" in out and "[intercepted]" in out, out


def test_an_intercepted_call_counted_by_its_sub_call_line_gets_no_second_count():
    out = render_text(build_interactive_badge(_metrics(_event("pandas.concat", "pd.concat([a, b])"))))
    assert "sub-call pd.concat([a, b]): 0/1 hit" in out, out
    assert "0/1 cached" not in out, out
    assert "counted in the sub-call lines above" in out, out


def test_a_decorated_call_keeps_its_heading_and_count():
    out = render_text(build_interactive_badge(_metrics(_event("__main__.fit", None, intercepted=False, hit=True))))
    assert "@cash.cache:" in out and "fit(): 1/1 cached" in out, out
    assert "intercepted calls:" not in out, out


def test_decorated_and_intercepted_calls_each_get_their_heading():
    out = render_text(
        build_interactive_badge(
            _metrics(
                _event("__main__.fit", None, intercepted=False),
                _event("pandas.concat", "pd.concat([a, b])"),
            )
        )
    )
    decorated, intercepted = out.index("@cash.cache:"), out.index("intercepted calls:")
    assert out.index("fit()") > decorated and out.index("concat()") > intercepted, out


def test_a_call_is_counted_by_a_sub_call_line_of_its_own_name_only():
    """``mid`` is not counted by ``sub-call pyramid(3)``: its name only ends
    the other one's."""
    out = render_text(build_interactive_badge(_metrics(_event("m.pyramid", "pyramid(3)"), _event("m.mid", "f(2)"))))
    assert "pyramid() [intercepted]: counted in the sub-call lines above" in out, out
    assert "mid() [intercepted]: 0/1 cached" in out, out


def test_the_html_badge_gives_each_kind_its_section():
    badge = build_interactive_badge(
        _metrics(_event("__main__.fit", None, intercepted=False), _event("pandas.concat", "pd.concat([a, b])"))
    )
    sections = {s.header: [g.func_name for g in s.items] for s in badge.sections if s.kind is SectionKind.DECORATORS}
    assert sections == {"DECORATOR CACHE (@cash.cache)": ["__main__.fit"], "INTERCEPTED CALLS": ["pandas.concat"]}
