"""An intercepted call must be legible as such on the badge (CAS-243).

Intercepted calls land in the same "N calls, M cached" region as hand-decorated
ones, which is the right place — it is the same cache. But undifferentiated it
misleads in both directions:

- a user who decorated nothing sees a ``@cash.cache`` section appear and cannot
  tell where it came from;
- a user who wrote ``# @cash:cache-calls`` has no way to confirm it engaged,
  which the directive's own docs tell them to do.

So the group carries the distinction and the renderers show it.
"""
from __future__ import annotations

import types

import pytest

import cash
from cash.notebook.badge_renderer.view_builder import build_interactive_badge
from cash.notebook.badge_renderer.renderers.text import render_text
from cash.notebook.call_interception import CallCache, CallSite


# ---------------------------------------------------------------- CallCache

@pytest.fixture
def call_cache(tmp_path):
    return CallCache(cash.Cash(cache_dir=str(tmp_path / "cc")))


def _site(source="compute(x)", names=("compute", "x")):
    return CallSite(
        source=source, free_names=frozenset(names), occurrence_index=0,
        computed_arg_positions=(0,),
    )


def test_wrapped_names_records_what_it_intercepted(call_cache):
    """The processor needs a key it can match drained log entries against."""
    def compute(x):
        return x + 1

    call_cache.set_sites([_site()])
    assert call_cache.wrapped_names == set()
    call_cache.resolve(compute)
    assert any(n.endswith("compute") for n in call_cache.wrapped_names), (
        f"expected compute's qualified name, got {call_cache.wrapped_names}"
    )


def test_passed_through_callables_are_not_recorded(call_cache):
    """Only what was actually wrapped counts, or the badge over-claims."""
    call_cache.set_sites([_site(source="len(a)", names=("len", "a"))])
    call_cache.resolve(len)
    call_cache.resolve(None)
    assert call_cache.wrapped_names == set()


def test_already_decorated_functions_are_not_recorded(call_cache, tmp_path):
    """A hand-decorated call must keep reading as hand-decorated."""
    other = cash.Cash(cache_dir=str(tmp_path / "other"))

    @other.cache
    def compute(x):
        return x + 1

    call_cache.set_sites([_site()])
    call_cache.resolve(compute)
    assert call_cache.wrapped_names == set()


# ------------------------------------------------------------- rendering

def _metrics(intercepted: bool):
    return [{
        "status": "COMPUTED",
        "code": "out.append(compute(x))",
        "total_time": 0.4,
        "evaluated_vars": ["out"],
        "decorator_calls": [
            {"func_name": "__main__.compute", "cache_hit": True,
             "execution_time": 0.001, "intercepted": intercepted},
            {"func_name": "__main__.compute", "cache_hit": False,
             "execution_time": 0.2, "intercepted": intercepted},
        ],
        "is_upstream": False,
    }]


def test_text_badge_marks_an_intercepted_group():
    text = render_text(build_interactive_badge(_metrics(intercepted=True)))
    assert "compute()" in text, text
    assert "cache-calls" in text, (
        f"an intercepted group must name the directive that produced it:\n{text}"
    )


def test_text_badge_leaves_a_decorated_group_unmarked():
    """Positive control: the marker must not appear for ordinary decorated calls."""
    text = render_text(build_interactive_badge(_metrics(intercepted=False)))
    assert "compute()" in text, text
    assert "cache-calls" not in text, (
        f"a hand-decorated group was mislabelled as intercepted:\n{text}"
    )


def _html(intercepted: bool, n_calls: int = 2) -> str:
    from cash.notebook.badge_renderer.renderers.html import render_html
    metrics = _metrics(intercepted)
    call = metrics[0]["decorator_calls"][0]
    metrics[0]["decorator_calls"] = [dict(call) for _ in range(n_calls)]
    return render_html(build_interactive_badge(metrics))


def test_html_badge_marks_an_intercepted_group():
    """A notebook shows HTML badges by default, so the marker must reach there too."""
    assert "@cache-calls" in _html(intercepted=True), (
        "the HTML badge does not distinguish an intercepted call"
    )


def test_html_badge_marks_an_intercepted_group_when_condensed():
    """The >3-call path renders the group directly and must mark it too.

    The two HTML paths are separate code: below the condense threshold each
    call renders its own row and never sees the group, so marking only the
    condensed branch would leave short loops unlabelled.
    """
    assert "@cache-calls" in _html(intercepted=True, n_calls=5)


def test_html_badge_leaves_decorated_groups_unmarked():
    """Positive control, both paths."""
    assert "@cache-calls" not in _html(intercepted=False, n_calls=2)
    assert "@cache-calls" not in _html(intercepted=False, n_calls=5)


def test_absent_flag_reads_as_decorated():
    """Metrics from before this field existed must not start claiming interception."""
    metrics = _metrics(intercepted=False)
    for call in metrics[0]["decorator_calls"]:
        del call["intercepted"]
    text = render_text(build_interactive_badge(metrics))
    assert "cache-calls" not in text, text
